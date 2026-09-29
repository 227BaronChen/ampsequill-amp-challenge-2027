"""Pinned ESMC construction and audited LoRA attachment."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch

EXPECTED_WEIGHT_SHA256 = (
    "323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea"
)
MODEL_REVISION = "7f10b20ae75017b2dbc884070e03434515709a8d"
TARGET_SUFFIXES = ["layernorm_qkv.1", "out_proj"]
TARGET_FULL_PATTERN = (
    r"^transformer\.blocks\.\d+\.attn\.(?:layernorm_qkv\.1|out_proj)$"
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def configure_v100_determinism(seed: int, device: str = "cuda:0") -> None:
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        # CPU runs are useful for smoke testing and are deterministic within a
        # fixed thread/layout contract, but are not assumed byte-identical to
        # the canonical V100 FP16 run.
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_base_model(
    weights: Path,
    expected_sha256: str = EXPECTED_WEIGHT_SHA256,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.nn.Module, object, dict[str, Any]]:
    actual_sha256 = sha256_file(weights)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"weight SHA256 mismatch: {actual_sha256} != {expected_sha256}")
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    tokenizer = get_esmc_model_tokenizers()
    model = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=tokenizer,
        use_flash_attn=False,
    )
    state_dict = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model = model.to(device=device, dtype=dtype)
    metadata = {
        "repo_id": "biohub/esmc-300m-2024-12",
        "revision": MODEL_REVISION,
        "weight_path": str(weights),
        "weight_sha256": actual_sha256,
        "weight_size_bytes": weights.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, tokenizer, metadata


def enumerate_lora_targets(model: torch.nn.Module) -> dict[str, Any]:
    matcher = re.compile(TARGET_FULL_PATTERN)
    linear_names = [
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
    ]
    matched = [name for name in linear_names if matcher.fullmatch(name)]
    unexpected = [name for name in matched if ".attn." not in name]
    forbidden = [
        name
        for name in matched
        if name.startswith("sequence_head")
        or name == "embed"
        or "lm_head" in name
        or "output" in name
    ]
    expected_count = 30 * 2
    if len(matched) != expected_count:
        raise RuntimeError(f"expected {expected_count} LoRA targets, got {len(matched)}")
    if unexpected or forbidden:
        raise RuntimeError(
            f"unsafe LoRA target match: unexpected={unexpected}, forbidden={forbidden}"
        )
    return {
        "target_modules": list(TARGET_SUFFIXES),
        "full_target_pattern": TARGET_FULL_PATTERN,
        "matched_count": len(matched),
        "matched_modules": matched,
        "examples": matched[:4] + matched[-4:],
        "forbidden_matches": forbidden,
        "all_linear_module_count": len(linear_names),
    }


def attach_lora(
    model: torch.nn.Module,
    rank: int,
    alpha: int | None = None,
    dropout: float = 0.0,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model

    audit = enumerate_lora_targets(model)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else rank * 2,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(TARGET_SUFFIXES),
    )
    peft_model = get_peft_model(model, config)
    trainable = sum(
        parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in peft_model.parameters())
    actual_trainable_names = [
        name for name, parameter in peft_model.named_parameters() if parameter.requires_grad
    ]
    if not actual_trainable_names or any("lora_" not in name for name in actual_trainable_names):
        raise RuntimeError("non-LoRA parameter was left trainable")
    audit.update(
        {
            "rank": rank,
            "alpha": config.lora_alpha,
            "dropout": dropout,
            "trainable_parameter_count": trainable,
            "total_parameter_count": total,
            "trainable_fraction": trainable / total,
            "trainable_tensor_count": len(actual_trainable_names),
        }
    )
    return peft_model, audit


def load_lora_adapter(
    model: torch.nn.Module, adapter_dir: Path
) -> torch.nn.Module:
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
