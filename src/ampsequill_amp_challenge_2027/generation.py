"""Deterministic iterative unmasking keyed by global sample ID."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import torch

CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


POSITION_STRATEGIES = (
    "sampled_token_probability",
    "entropy",
    "random",
)


def canonical_token_ids(tokenizer: object) -> list[int]:
    ids = [int(tokenizer.convert_tokens_to_ids(amino_acid)) for amino_acid in CANONICAL_AMINO_ACIDS]
    if len(set(ids)) != 20 or any(token_id < 0 for token_id in ids):
        raise ValueError(f"tokenizer does not provide 20 unique canonical amino acids: {ids}")
    return ids


def empirical_length_counts(sequences: Iterable[str]) -> dict[int, int]:
    return dict(sorted(Counter(map(len, sequences)).items()))


def choose_length(
    length_counts: dict[int, int], generator: torch.Generator
) -> int:
    lengths = sorted(length_counts)
    weights = torch.tensor([length_counts[length] for length in lengths], dtype=torch.float64)
    index = int(torch.multinomial(weights, 1, generator=generator).item())
    return lengths[index]


def sample_seed(base_seed: int, global_sample_id: int) -> int:
    # Keep each sample's RNG stream independent. Model logits may still vary when
    # the GPU forward layout changes, so exact output reproducibility additionally
    # requires the released batch layout documented in the README.
    return (base_seed * 1_000_003 + global_sample_id * 97_409 + 17) % (2**63 - 1)


def position_seed(base_seed: int, global_sample_id: int) -> int:
    # Use a disjoint deterministic stream so position selection never consumes token RNG.
    return sample_seed(base_seed + 2_000_000, global_sample_id)


def top_p_sample(
    logits: torch.Tensor,
    generator: torch.Generator,
    top_p: float,
) -> tuple[int, float]:
    probabilities = torch.softmax(logits.float().cpu(), dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative - sorted_probabilities >= top_p
    sorted_probabilities[remove] = 0.0
    sorted_probabilities /= sorted_probabilities.sum()
    sampled_rank = int(torch.multinomial(sorted_probabilities, 1, generator=generator).item())
    token_id = int(sorted_indices[sampled_rank].item())
    confidence = float(probabilities[token_id].item())
    return token_id, confidence


def canonical_entropy(
    raw_logits: torch.Tensor,
    canonical_ids: list[int],
) -> float:
    probabilities = torch.softmax(raw_logits[canonical_ids].float().cpu(), dim=-1)
    positive = probabilities > 0
    return float(-(probabilities[positive] * probabilities[positive].log()).sum().item())


def select_commit_positions(
    *,
    unresolved_positions: list[int],
    sampled_candidates: list[tuple[float, int, int]],
    raw_logits: torch.Tensor,
    canonical_ids: list[int],
    position_strategy: str,
    position_generator: torch.Generator | None,
    new_count: int,
) -> list[tuple[int, int]]:
    if position_strategy not in POSITION_STRATEGIES:
        raise ValueError(
            f"unknown position_strategy {position_strategy!r}; expected one of "
            f"{POSITION_STRATEGIES}"
        )
    if len(sampled_candidates) != len(unresolved_positions):
        raise ValueError("sampled candidates and unresolved positions differ")
    if not 0 <= new_count <= len(unresolved_positions):
        raise ValueError("new_count is outside the unresolved position range")
    sampled_by_position = {
        position: token_id for _, position, token_id in sampled_candidates
    }
    if position_strategy == "sampled_token_probability":
        ordered = sorted(sampled_candidates, key=lambda item: (-item[0], item[1]))
        return [(position, token_id) for _, position, token_id in ordered[:new_count]]
    if position_strategy == "entropy":
        ordered_positions = sorted(
            unresolved_positions,
            key=lambda position: (
                canonical_entropy(raw_logits[position], canonical_ids),
                position,
            ),
        )
    else:
        if position_generator is None:
            raise ValueError("random position strategy requires a position generator")
        permutation = torch.randperm(
            len(unresolved_positions),
            generator=position_generator,
            device="cpu",
        ).tolist()
        ordered_positions = [unresolved_positions[index] for index in permutation]
    return [
        (position, sampled_by_position[position])
        for position in ordered_positions[:new_count]
    ]


@torch.inference_mode()
def iterative_unmask_batch(
    model: torch.nn.Module,
    tokenizer: object,
    lengths: list[int],
    global_sample_ids: list[int],
    base_seed: int,
    steps: int,
    temperature_start: float,
    temperature_end: float,
    top_p: float,
    device: str = "cuda:0",
    position_strategy: str = "sampled_token_probability",
    position_schedule_sink: list[dict[str, Any]] | None = None,
) -> list[str]:
    if len(lengths) != len(global_sample_ids):
        raise ValueError("lengths and global_sample_ids must have the same size")
    if steps < 1:
        raise ValueError("steps must be positive")
    if position_strategy not in POSITION_STRATEGIES:
        raise ValueError(
            f"unknown position_strategy {position_strategy!r}; expected one of "
            f"{POSITION_STRATEGIES}"
        )
    batch_size = len(lengths)
    maximum_length = max(lengths)
    cls_id = int(tokenizer.cls_token_id)
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)
    canonical_ids = canonical_token_ids(tokenizer)
    input_ids = torch.full(
        (batch_size, maximum_length + 2),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    residue_positions = torch.zeros_like(input_ids, dtype=torch.bool)
    input_ids[:, 0] = cls_id
    for row, length in enumerate(lengths):
        input_ids[row, 1 : length + 1] = mask_id
        input_ids[row, length + 1] = eos_id
        residue_positions[row, 1 : length + 1] = True
    token_generators = [
        torch.Generator(device="cpu").manual_seed(sample_seed(base_seed, sample_id))
        for sample_id in global_sample_ids
    ]
    position_generators = [
        torch.Generator(device="cpu").manual_seed(position_seed(base_seed, sample_id))
        for sample_id in global_sample_ids
    ]
    position_schedules: list[list[list[int]]] = [[] for _ in global_sample_ids]
    committed = torch.zeros_like(residue_positions)
    allowed = torch.full((64,), float("-inf"), dtype=torch.float32, device=device)
    allowed[canonical_ids] = 0.0
    model.eval()
    for step_index in range(steps):
        fraction = (step_index + 1) / steps
        temperature = temperature_start + (
            temperature_end - temperature_start
        ) * fraction
        output = model(sequence_tokens=input_ids)
        raw_logits = (output.sequence_logits.float() + allowed.view(1, 1, -1)).cpu()
        sampling_logits = raw_logits / temperature
        for row, length in enumerate(lengths):
            unresolved = (residue_positions[row] & ~committed[row]).nonzero(
                as_tuple=False
            ).flatten()
            if unresolved.numel() == 0:
                continue
            candidates: list[tuple[float, int, int]] = []
            for position in unresolved.tolist():
                token_id, confidence = top_p_sample(
                    sampling_logits[row, position], token_generators[row], top_p
                )
                candidates.append((confidence, position, token_id))
            target_committed = math.ceil(length * fraction)
            new_count = min(
                len(candidates),
                max(1, target_committed - int(committed[row].sum().item())),
            )
            selected = select_commit_positions(
                unresolved_positions=unresolved.tolist(),
                sampled_candidates=candidates,
                raw_logits=raw_logits[row],
                canonical_ids=canonical_ids,
                position_strategy=position_strategy,
                position_generator=position_generators[row],
                new_count=new_count,
            )
            position_schedules[row].append(
                [position - 1 for position, _ in selected]
            )
            for position, token_id in selected:
                input_ids[row, position] = token_id
                committed[row, position] = True
    if not bool((committed == residue_positions).all().item()):
        raise RuntimeError("iterative unmasking left unresolved residue positions")
    sequences: list[str] = []
    for row, length in enumerate(lengths):
        tokens = input_ids[row, 1 : length + 1].tolist()
        sequence = "".join(str(tokenizer.convert_ids_to_tokens(token)) for token in tokens)
        if len(sequence) != length or set(sequence) - set(CANONICAL_AMINO_ACIDS):
            raise RuntimeError(f"invalid generated sequence: {sequence!r}")
        sequences.append(sequence)
    if position_schedule_sink is not None:
        position_schedule_sink.extend(
            {
                "global_sample_id": sample_id,
                "position_strategy": position_strategy,
                "committed_residue_positions_by_step": schedule,
            }
            for sample_id, schedule in zip(
                global_sample_ids,
                position_schedules,
                strict=True,
            )
        )
    return sequences
