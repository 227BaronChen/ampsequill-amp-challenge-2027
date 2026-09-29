"""Check the materialized AMPsequill FASTA files without running ESM-C."""
from __future__ import annotations

import json
from pathlib import Path

import Levenshtein

from ampsequill_amp_challenge_2027.selection import (
    EXPECTED_LIBRARY_FASTA_SHA256,
    EXPECTED_REFERENCE_SHA256,
    library_fasta_sha256,
    load_cache,
    select_top_candidates,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def fasta(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) % 2 or not all(lines[i].startswith(">") for i in range(0, len(lines), 2)):
        raise ValueError(f"invalid two-line FASTA: {path}")
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]


def main() -> None:
    manifest = json.loads((ROOT / "generate/submission-manifest.json").read_text())
    library_path = ROOT / "generate/library.fasta"
    top_path = ROOT / "generate/top.fasta"
    reference_path = ROOT / "data/antibacterial.fasta"
    if sha256(library_path) != EXPECTED_LIBRARY_FASTA_SHA256:
        raise ValueError("library FASTA hash mismatch")
    if sha256(reference_path) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("public reference hash mismatch")
    if sha256(top_path) != manifest["materialized_outputs"]["top_fasta_sha256"]:
        raise ValueError("top FASTA hash mismatch")
    library = fasta(library_path)
    top = fasta(top_path)
    if len(library) != 50_000 or len(top) != 100:
        raise ValueError("library or Top-100 count mismatch")
    library_sequences = [sequence for _, sequence in library]
    top_sequences = [sequence for _, sequence in top]
    if len(set(library_sequences)) != 50_000 or len(set(top_sequences)) != 100:
        raise ValueError("duplicate sequence")
    if not set(top_sequences) <= set(library_sequences):
        raise ValueError("Top-100 is not a library subset")
    if any(not 8 <= len(s) <= 50 or set(s) - CANONICAL for s in library_sequences):
        raise ValueError("invalid library sequence")
    if any(not 8 <= len(s) <= 40 or set(s) - CANONICAL for s in top_sequences):
        raise ValueError("invalid Top-100 sequence")
    references = [sequence for _, sequence in fasta(reference_path)]
    if set(library_sequences) & set(references):
        raise ValueError("library has an exact public reference match")
    for rank, sequence in enumerate(top_sequences):
        if max(Levenshtein.ratio(sequence, reference) for reference in references) > 0.8:
            raise ValueError(f"Top-100 reference similarity violation: rank {rank + 1}")
        if any(Levenshtein.ratio(sequence, earlier) >= 0.65 for earlier in top_sequences[:rank]):
            raise ValueError(f"Top-100 pairwise diversity violation: rank {rank + 1}")
    if any(header != f">seq{rank:05d}" for rank, (header, _) in enumerate(library, 1)):
        raise ValueError("library FASTA headers changed")
    # Recover sample IDs from the immutable cache. load_cache verifies IDs and sequences.
    import gzip
    rows = []
    with gzip.open(ROOT / "data/selection_scores.jsonl.gz", "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            cached = json.loads(line)
            rows.append({"accepted_index": index,
                         "global_sample_id": cached["global_sample_id"],
                         "sequence": library_sequences[index]})
    if library_fasta_sha256(rows) != EXPECTED_LIBRARY_FASTA_SHA256:
        raise ValueError("library formatting changed")
    selected, pool_count = select_top_candidates(load_cache(ROOT / "data/selection_scores.jsonl.gz", rows))
    expected_headers = [f">accepted_index={row['accepted_index']} global_sample_id={row['global_sample_id']}"
                        for row in selected]
    if top != list(zip(expected_headers, [row["sequence"] for row in selected])):
        raise ValueError("Top-100 differs from the replayed four-scenario ranking")
    print(json.dumps({"status": "verified", "library": len(library), "top": len(top),
                      "eligible_pool": pool_count, "library_sha256": sha256(library_path),
                      "top_sha256": sha256(top_path)}, indent=2))


if __name__ == "__main__":
    main()
