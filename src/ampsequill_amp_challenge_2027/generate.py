"""Deterministic ESMC-LoRA AMP Challenge 2027 generator.

The default path generates the submitted library and ranks its top candidates
with a hash-checked score cache. CPU execution is supported for diagnostic
platform checks; the release platform is a single V100 using FP16 and SDPA
math attention.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .generation import choose_length, empirical_length_counts, iterative_unmask_batch, sample_seed
from .modeling import EXPECTED_WEIGHT_SHA256, load_base_model, load_lora_adapter, configure_v100_determinism
from .xamp_models import model_XAMP
from .selection import (
    EXPECTED_LIBRARY_FASTA_SHA256,
    EXPECTED_REFERENCE_SHA256,
    library_fasta_sha256,
    load_cache,
    select_top_candidates,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
MODEL_REPO = "biohub/esmc-300m-2024-12"
MODEL_REVISION = "7f10b20ae75017b2dbc884070e03434515709a8d"
MODEL_FILENAME = "data/weights/esmc_300m_2024_12_v0.pth"
XAMP_T_SHA256 = "7a212e70b4e522a500cf5bffd134375cab9e9033223dcfead86adacbe69ed07d"
XAMP_E_SHA256 = "093fb31e519c629d8de8ad04c8a2ed29169590267ee598c4a449dccb81a8fd2e"
ADAPTER_SHA256 = "104168df75fa28841acf7fc16446200044aabded2813b16ff0611fe2c31f5e23"
LENGTH_COUNTS_SHA256 = "31b91725f0a8a1d2b6b44eb7cbc238c00b364e00313b1df24d437b0ddfccd752"
KNOWN_HASHES_SHA256 = "390b1db54fab6d3d8171829dd4a541dc7fff61aaa54b771a30b045c357877d88"
OFFICIAL_REFERENCE_SHA256 = "cbbeac64ba95746d87961e8ad9dd0849ae8058d15a300b2e7f6990730ca521e9"
ESM2_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"
ESM2_MODEL_SHA256 = "0b532c25b8b6debba372981cecb98c11a8639b9da833d2dc2302bdc446cc4a1f"
APEX_COLUMNS = (
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
)


class _HFESM2Classifier(torch.nn.Module):
    """XAMP-E head backed by the public Hugging Face ESM-2 implementation.

    The original XAMP checkpoint stores the ESM-2 base weights under the
    fair-esm module names.  The public ``esm`` package required for ESMC does
    not expose that legacy API, so the release uses the equivalent public
    Transformers ESM-2 implementation and loads the frozen XAMP MLP head.
    """

    def __init__(self, model_id_or_path: str | Path, revision: str | None = None):
        super().__init__()
        from transformers import AutoTokenizer, EsmModel

        kwargs = {"revision": revision} if revision and not Path(str(model_id_or_path)).is_dir() else {}
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_id_or_path), **kwargs)
        self.esm_model = EsmModel.from_pretrained(
            str(model_id_or_path), add_pooling_layer=False, **kwargs
        )
        hidden = int(self.esm_model.config.hidden_size)
        if hidden != 480:
            raise RuntimeError(f"XAMP-E expects ESM2 hidden size 480, got {hidden}")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
        )
        for parameter in self.esm_model.parameters():
            parameter.requires_grad = False

    def load_xamp_state(self, state: dict[str, Any]) -> None:
        head = {
            key.removeprefix("mlp."): value
            for key, value in state.items()
            if key.startswith("mlp.")
        }
        if len(head) != len(self.mlp.state_dict()):
            raise RuntimeError("XAMP-E classifier head is incomplete")
        self.mlp.load_state_dict(head, strict=True)

    def forward(self, sequences: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = self.esm_model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].to(dtype=output.dtype)
        representation = (output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        )
        return self.mlp(representation)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> list[str]:
    values: list[str] = []
    current: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(">"):
            if current:
                values.append("".join(current).upper())
                current = []
        elif line.strip():
            current.append(line.strip())
    if current:
        values.append("".join(current).upper())
    return values


def resolve_weights() -> Path:
    override = os.environ.get("ESMC_WEIGHTS")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ESMC_WEIGHTS does not exist: {path}")
        return path
    # Standard ranged HTTP is more portable than the optional Xet client on
    # restricted/proxied competition machines.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to fetch the pinned ESMC base model; "
            "set ESMC_WEIGHTS to a verified local file for offline use"
        ) from exc
    path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
        )
    )
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_length_counts(path: Path) -> dict[int, int]:
    if sha256_file(path) != LENGTH_COUNTS_SHA256:
        raise RuntimeError("length_counts.json SHA256 does not match the frozen dataset")
    values = load_json(path)
    result = {int(key): int(value) for key, value in values.items()}
    if set(result) != set(range(8, 51)) or sum(result.values()) != 22262:
        raise RuntimeError("length_counts.json does not match the disclosed training set")
    return result


def load_known_hashes(path: Path) -> set[str]:
    if sha256_file(path) != KNOWN_HASHES_SHA256:
        raise RuntimeError("known_sequence_sha256.txt SHA256 does not match the frozen dataset")
    hashes = {line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()}
    if len(hashes) != 22262 or any(len(value) != 64 for value in hashes):
        raise RuntimeError("known_sequence_sha256.txt is malformed")
    return hashes


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def canonical_json_line(value: Any) -> bytes:
    """Return the byte representation used by the frozen JSONL artifacts."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def decide_attempt(
    sequence: str,
    accepted: set[str],
    known_hashes: set[str],
    official: set[str],
) -> str | None:
    if not (8 <= len(sequence) <= 50) or set(sequence) - set(CANONICAL):
        return "invalid_or_noncanonical"
    if sequence in accepted:
        return "duplicate_within_current_pool"
    if sequence_hash(sequence) in known_hashes:
        return "training_set_exact_match"
    if sequence in official:
        return "official_reference_exact_known"
    return None


def _load_generation_model(
    weights: Path, device: str, cpu_dtype: str = "auto"
) -> tuple[torch.nn.Module, object, str]:
    if cpu_dtype not in {"auto", "fp16", "fp32"}:
        raise ValueError("cpu_dtype must be auto, fp16, or fp32")
    # The external CPU audit found FP16 operators to be functionally valid but
    # pathologically slow (roughly one effective core).  Keep explicit FP16
    # available for diagnostics, while making the safe ``auto`` CPU default
    # use the validated FP32 path.
    dtype = (
        torch.float32
        if device.startswith("cpu") and cpu_dtype in {"auto", "fp32"}
        else torch.float16
    )
    try:
        model, tokenizer, _ = load_base_model(weights, device=device, dtype=dtype)
    except (RuntimeError, ValueError) as exc:
        if not device.startswith("cpu"):
            raise
        print(f"[generate] CPU FP16 unavailable; retrying FP32: {exc}", file=sys.stderr)
        model, tokenizer, _ = load_base_model(weights, device=device, dtype=torch.float32)
        dtype = torch.float32
    adapter = ROOT / "checkpoint" / "lora"
    if sha256_file(adapter / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise RuntimeError("LoRA adapter SHA256 does not match the frozen checkpoint")
    model = load_lora_adapter(model, adapter).eval()
    return model, tokenizer, str(dtype).replace("torch.", "")


def generate_library(
    *,
    model: torch.nn.Module,
    tokenizer: object,
    device: str,
    seed: int,
    target: int,
    batch_size: int,
    top_p: float = 0.98,
    steps: int = 8,
    temperature_start: float = 1.2,
    temperature_end: float = 0.7,
    diagnostics_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = load_length_counts(ROOT / "data" / "length_counts.json")
    known_hashes = load_known_hashes(ROOT / "data" / "known_sequence_sha256.txt")
    official_path = ROOT / "data" / "antibacterial.fasta"
    if sha256_file(official_path) != OFFICIAL_REFERENCE_SHA256:
        raise RuntimeError("official reference FASTA SHA256 does not match the frozen input")
    official = set(read_fasta(official_path))
    accepted: set[str] = set()
    rows: list[dict[str, Any]] = []
    raw_attempts = 0
    attempted_lengths: list[int] = []
    position_records: list[dict[str, Any]] = []
    diagnostics_handles: tuple[Any, Any] | None = None
    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_handles = (
            (diagnostics_dir / "raw_generation.jsonl").open(
                "wb"
            ),
            (diagnostics_dir / "position_schedule.jsonl").open(
                "wb"
            ),
        )
    max_attempts = max(batch_size, int(math.ceil(target * 1.28 / batch_size)) * batch_size)
    try:
        while raw_attempts < max_attempts and len(rows) < target:
            ids = list(range(raw_attempts, raw_attempts + batch_size))
            lengths: list[int] = []
            for sample_id in ids:
                generator = torch.Generator(device="cpu").manual_seed(
                    sample_seed(seed + 1_000_000, sample_id)
                )
                lengths.append(choose_length(counts, generator))
            attempted_lengths.extend(lengths)
            schedules: list[dict[str, Any]] = []
            sequences = iterative_unmask_batch(
                model=model,
                tokenizer=tokenizer,
                lengths=lengths,
                global_sample_ids=ids,
                base_seed=seed,
                steps=steps,
                temperature_start=temperature_start,
                temperature_end=temperature_end,
                top_p=top_p,
                device=device,
                position_strategy="random",
                position_schedule_sink=schedules,
            )
            position_records.extend(schedules)
            if diagnostics_handles is not None:
                raw_handle, position_handle = diagnostics_handles
                for schedule in schedules:
                    position_handle.write(canonical_json_line(schedule))
                for sample_id, length, sequence in zip(
                    ids, lengths, sequences, strict=True
                ):
                    raw_handle.write(
                        canonical_json_line(
                            {
                                "global_sample_id": sample_id,
                                "length": length,
                                "sequence": sequence,
                            }
                        )
                    )
                raw_handle.flush()
                position_handle.flush()
            for sample_id, length, sequence in zip(ids, lengths, sequences, strict=True):
                if len(rows) >= target:
                    continue
                reason = decide_attempt(sequence, accepted, known_hashes, official)
                if reason is None:
                    accepted.add(sequence)
                    rows.append(
                        {
                            "accepted_index": len(rows),
                            "global_sample_id": sample_id,
                            "length": length,
                            "sequence": sequence,
                        }
                    )
            raw_attempts += batch_size
            if raw_attempts % (batch_size * 16) == 0 or len(rows) >= target:
                print(
                    f"[generate] {len(rows)}/{target} accepted after "
                    f"{raw_attempts} attempts",
                    flush=True,
                )
    finally:
        if diagnostics_handles is not None:
            for handle in diagnostics_handles:
                handle.close()
    if len(rows) != target:
        raise RuntimeError(f"raw-attempt ceiling reached with {len(rows)}/{target} accepted")
    position_bytes = b"".join(
        canonical_json_line(record)
        for record in sorted(position_records, key=lambda value: value["global_sample_id"])
    )
    length_bytes = "\n".join(map(str, attempted_lengths)).encode("ascii") + b"\n"
    return rows, {
        "raw_attempts": raw_attempts,
        "target": target,
        "batch_size": batch_size,
        "seed": seed,
        "steps": steps,
        "position_strategy": "random",
        "top_p": top_p,
        "temperature_start": temperature_start,
        "temperature_end": temperature_end,
        "raw_attempt_length_schedule_sha256": hashlib.sha256(length_bytes).hexdigest(),
        "position_schedule_sha256": hashlib.sha256(position_bytes).hexdigest(),
        "raw_attempt_count": len(attempted_lengths),
        "position_schedule_prefix": [
            record
            for record in sorted(position_records, key=lambda value: value["global_sample_id"])
            if int(record["global_sample_id"]) < 64
        ],
    }


def load_torch_state(path: Path, device: torch.device) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return torch.load(handle, map_location=device, weights_only=False)


def load_xamp_e_model(device: torch.device) -> _HFESM2Classifier:
    model_path = ROOT / "checkpoint" / "xamp" / "esm2_t12_35M_UR50D"
    if not model_path.is_dir():
        override = os.environ.get("XAMP_ESM2_MODEL")
        model_path = Path(override).expanduser() if override else "facebook/esm2_t12_35M_UR50D"
    if isinstance(model_path, Path) and model_path.is_dir():
        local_weights = model_path / "model.safetensors"
        if sha256_file(local_weights) != ESM2_MODEL_SHA256:
            raise RuntimeError("XAMP-E ESM-2 base SHA256 does not match the manifest")
        model = _HFESM2Classifier(model_path).to(device)
    else:
        model = _HFESM2Classifier(model_path, revision=ESM2_REVISION).to(device)
    state = load_torch_state(
        ROOT / "checkpoint" / "xamp" / "xamp_e.state_dict.pth.gz", device
    )
    model.load_xamp_state(state)
    return model


def xamp_scores(sequences: list[str], device: str, cpu_threads: int) -> list[dict[str, float]]:
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(42)
    target = torch.device(device)
    xamp_dir = ROOT / "checkpoint" / "xamp"
    expected = {
        "xamp_t.state_dict.pth.gz": XAMP_T_SHA256,
        "xamp_e.state_dict.pth.gz": XAMP_E_SHA256,
    }
    for filename, digest in expected.items():
        path = xamp_dir / filename
        if sha256_file(path) != digest:
            raise RuntimeError(f"XAMP asset SHA256 mismatch: {filename}")
    t_model = model_XAMP().to(target)
    t_model.load_state_dict(load_torch_state(xamp_dir / "xamp_t.state_dict.pth.gz", target))
    t_model.eval()
    e_model = load_xamp_e_model(target)
    e_model.eval()
    values: list[dict[str, float]] = []
    batch_size = 256
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        if target.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                t = t_model(batch).squeeze(-1).float().cpu().tolist()
                e = torch.sigmoid(e_model(batch).squeeze(-1)).float().cpu().tolist()
        else:
            with torch.inference_mode():
                t = t_model(batch).squeeze(-1).float().cpu().tolist()
                e = torch.sigmoid(e_model(batch).squeeze(-1)).float().cpu().tolist()
        for tv, ev in zip(t, e, strict=True):
            if not (math.isfinite(tv) and math.isfinite(ev) and 0 <= tv <= 1 and 0 <= ev <= 1):
                raise RuntimeError("XAMP returned a non-finite or out-of-range probability")
            values.append(
                {
                    "xamp_t_probability": float(tv),
                    "xamp_e_probability": float(ev),
                    "xamp_integrated_probability": float(min(tv, ev)),
                }
            )
    return values


def apex_scores(sequences: list[str], device: str, cpu_threads: int) -> list[dict[str, Any]]:
    import importlib.util

    import numpy as np
    apex_root = ROOT / "third_party" / "apex"
    sys.path.insert(0, str(apex_root))
    from utils import make_vocab, onehot_encoding

    torch.set_num_threads(cpu_threads)
    target = torch.device(device)
    checkpoints = sorted((ROOT / "checkpoint" / "apex" / "APEX_pathogen_models").glob("APEX_*") )
    if len(checkpoints) != 8:
        raise RuntimeError(f"expected 8 APEX checkpoints, found {len(checkpoints)}")
    word2idx, _ = make_vocab()
    encoded = onehot_encoding(sequences, 52, word2idx)
    accumulator = np.zeros((len(sequences), len(APEX_COLUMNS)), dtype=np.float64)
    with torch.inference_mode():
        for checkpoint in checkpoints:
            model = torch.load(checkpoint, map_location=target, weights_only=False).to(target).eval()
            for start in range(0, len(sequences), 3000):
                batch = torch.as_tensor(encoded[start : start + 3000], dtype=torch.long, device=target)
                if target.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        transformed = model(batch).float().cpu().numpy()
                else:
                    transformed = model(batch).float().cpu().numpy()
                accumulator[start : start + len(batch)] += np.power(10.0, 6.0 - transformed, dtype=np.float64)
            del model
            if target.type == "cuda":
                torch.cuda.empty_cache()
    mic_values = accumulator / len(checkpoints)
    if not np.isfinite(mic_values).all() or not (mic_values > 0).all():
        raise RuntimeError("APEX produced invalid MIC values")
    rows: list[dict[str, Any]] = []
    for values in mic_values:
        arithmetic = float(np.mean(values))
        rows.append(
            {
                "mic_um": {name: float(value) for name, value in zip(APEX_COLUMNS, values)},
                "arithmetic_mean_mic_um": arithmetic,
                "geometric_mean_mic_um": float(np.exp(np.mean(np.log(values)))),
                "median_mic_um": float(np.median(values)),
                "worst_pathogen_mic_um": float(np.max(values)),
                "pathogen_count_mic_le_16": int(np.count_nonzero(values <= 16.0)),
                "pathogen_count_mic_le_32": int(np.count_nonzero(values <= 32.0)),
                "apex_broad_score": -arithmetic,
            }
        )
    return rows


def write_fasta(path: Path, rows: Iterable[dict[str, Any]], prefix: str) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for rank, row in enumerate(rows, 1):
            if prefix == "seq":
                header = f">seq{rank:05d}"
            elif prefix == "rank":
                # Preserve the frozen Top-100 artifact's byte-level headers.
                header = (
                    f">accepted_index={row['accepted_index']} "
                    f"global_sample_id={row['global_sample_id']}"
                )
            else:
                raise ValueError(f"unsupported FASTA prefix: {prefix!r}")
            handle.write(f"{header}\n{row['sequence']}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sequences", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument(
        "--cpu-dtype",
        choices=("auto", "fp16", "fp32"),
        default="auto",
        help="CPU ESMC dtype; auto uses the validated FP32 path (FP16 remains an explicit diagnostic option).",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generate")
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help="Optional directory for raw-generation and position-schedule evidence; "
        "not written by the public default run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_sequences <= 0 or args.top_k <= 0 or args.top_k > args.n_sequences:
        raise SystemExit("require 0 < --top-k <= --n-sequences")
    if args.batch_size != 64:
        raise SystemExit("batch size is frozen at 64 for reproducibility")
    if args.n_sequences != 50_000 or args.top_k != 100 or args.seed != 42:
        raise SystemExit("submission requires --n-sequences 50000 --top-k 100 --seed 42")
    if args.cpu_threads < 1:
        raise SystemExit("--cpu-threads must be positive")
    device = "cuda:0" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and device == "cpu":
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    output_dir = args.output_dir.resolve()
    canonical_output_dir = (ROOT / "generate").resolve()
    if device == "cpu" and (
        output_dir == canonical_output_dir or canonical_output_dir in output_dir.parents
    ):
        raise SystemExit(
            "CPU output must be isolated from the frozen submission directory; "
            "pass --output-dir cpu-full-a (or another external directory)"
        )
    torch.set_num_threads(args.cpu_threads if device == "cpu" else 1)
    configure_v100_determinism(args.seed, device=device)
    weights = resolve_weights()
    if sha256_file(weights) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("ESMC base weight SHA256 mismatch")
    started = time.monotonic()
    model, tokenizer, dtype = _load_generation_model(weights, device, args.cpu_dtype)
    rows, sampling = generate_library(
        model=model,
        tokenizer=tokenizer,
        device=device,
        seed=args.seed,
        target=args.n_sequences,
        batch_size=args.batch_size,
        diagnostics_dir=args.diagnostics_dir.resolve() if args.diagnostics_dir else None,
    )
    if library_fasta_sha256(rows) != EXPECTED_LIBRARY_FASTA_SHA256:
        raise RuntimeError("generated library differs from the submitted library")
    if sha256_file(ROOT / "data" / "antibacterial.fasta") != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("reference database differs from the submitted audit")
    cache = load_cache(ROOT / "data" / "selection_scores.jsonl.gz", rows)
    top, eligible_count = select_top_candidates(cache, args.top_k)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(output_dir / "library.fasta", rows, "seq")
    write_fasta(output_dir / "top.fasta", top, "rank")
    if args.diagnostics_dir:
        metadata_path = args.diagnostics_dir.resolve() / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "device": device,
                    "dtype": dtype,
                    "sampling": sampling,
                    "library_count": len(rows),
                    "top_count": len(top),
                    "selection": "four-scenario predicted-MIC ranking",
                    "eligible_count": eligible_count,
                    "library_sha256": sha256_file(output_dir / "library.fasta"),
                    "top_sha256": sha256_file(output_dir / "top.fasta"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "generated",
                "device": device,
                "dtype": dtype,
                "sampling": sampling,
                "library_count": len(rows),
                "top_count": len(top),
                "selection": "four-scenario predicted-MIC ranking",
                "eligible_count": eligible_count,
                "wall_time_seconds": time.monotonic() - started,
                "library_sha256": sha256_file(output_dir / "library.fasta"),
                "top_sha256": sha256_file(output_dir / "top.fasta"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
