"""Four-scenario selector for the submitted AMPsequill library.

The published score cache is tied to the submitted 50,000-sequence library.
A different library requires fresh scoring and validation.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import Levenshtein


THRESHOLDS = (8, 16, 32, 64)
EXPECTED_LIBRARY_FASTA_SHA256 = "d72633df797e392a4765cb42a819571ca56b7def5fb667d7bea8a533ad168274"
EXPECTED_REFERENCE_SHA256 = "cbbeac64ba95746d87961e8ad9dd0849ae8058d15a300b2e7f6990730ca521e9"
EXPECTED_CACHE_SHA256 = "b71da7c9939d16ded5bb0eeb3f195a7f1454b122eef4df2a1b3c2ef46c812033"
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
SPECIES = {
    "A. baumannii": "A. baumannii",
    "E. coli": "E. coli",
    "K. pneumoniae": "K. pneumoniae",
    "P. aeruginosa": "P. aeruginosa",
    "S. aureus": "S. aureus",
    "vancomycin-resistant E. faecalis": "E. faecalis",
    "vancomycin-resistant E. faecium": "E. faecium",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_fasta_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for rank, row in enumerate(rows, 1):
        digest.update(f">seq{rank:05d}\n{row['sequence']}\n".encode("ascii"))
    return digest.hexdigest()


def load_cache(path: Path, library_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(library_rows) != 50_000:
        raise ValueError("selection requires the submitted 50,000-sequence library")
    if sha256(path) != EXPECTED_CACHE_SHA256:
        raise ValueError("selection score cache SHA256 mismatch")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            original = library_rows[index] if index < len(library_rows) else None
            if original is None or any(row[key] != original[key] for key in
                                       ("accepted_index", "global_sample_id", "sequence")):
                raise ValueError(f"score cache / generated library mismatch at row {index}")
            rows.append(row)
    if len(rows) != len(library_rows):
        raise ValueError("score cache row count mismatch")
    return rows


def eligible(row: dict[str, Any]) -> bool:
    sequence = row["sequence"]
    hemo = row["hemopi2_hemolytic_score"]
    apex = row["apex_mic_um"]
    return (
        8 <= len(sequence) <= 40 and set(sequence) <= CANONICAL
        and not row["reference_exact"]
        and math.isfinite(float(row["reference_max_levenshtein_ratio"]))
        and float(row["reference_max_levenshtein_ratio"]) <= 0.8
        and math.isfinite(float(row["xamp_integrated_probability"]))
        and float(row["xamp_integrated_probability"]) >= 0.5
        and row["hemopi2_available"] and hemo is not None
        and math.isfinite(float(hemo)) and 0 <= float(hemo) < 0.46
        and len(apex) == 11
        and all(math.isfinite(float(value)) and float(value) > 0
                for value in apex.values())
    )


def _species(endpoint: str) -> str:
    for prefix, species in SPECIES.items():
        if endpoint.startswith(prefix):
            return species
    raise ValueError(f"unknown APEX endpoint: {endpoint}")


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.9
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _scenario_key(row: dict[str, Any], threshold: int) -> tuple[float, ...]:
    endpoints = row["apex_mic_um"]
    covered = {_species(name) for name, value in endpoints.items() if value <= threshold}
    values = list(endpoints.values())
    return (
        -len(covered),
        -sum(value <= threshold for value in values),
        _p90(values),
        statistics.median(values),
        int(row["accepted_index"]),
    )


def select_top_candidates(cache_rows: list[dict[str, Any]], top_k: int = 100) -> tuple[list[dict[str, Any]], int]:
    pool = [row for row in cache_rows if eligible(row)]
    if len(pool) != 5594:
        raise ValueError(f"eligible candidate pool changed: {len(pool)} != 5594")
    ranks = {
        threshold: {
            int(row["accepted_index"]): rank
            for rank, row in enumerate(sorted(pool, key=lambda row: _scenario_key(row, threshold)), 1)
        }
        for threshold in THRESHOLDS
    }
    ordered = sorted(pool, key=lambda row: (
        max(ranks[threshold][row["accepted_index"]] for threshold in THRESHOLDS),
        sum(ranks[threshold][row["accepted_index"]] for threshold in THRESHOLDS),
        row["accepted_index"],
    ))
    selected: list[dict[str, Any]] = []
    for row in ordered:
        if all(Levenshtein.ratio(row["sequence"], other["sequence"]) < 0.65
               for other in selected):
            selected.append(row)
            if len(selected) == top_k:
                return selected, len(pool)
    raise ValueError(f"only {len(selected)} candidates pass internal diversity")
