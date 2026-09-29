# AMPsequill — AMP Challenge 2027

![AMPsequill pixel-art banner](assets/ampsequill-pixel-banner.png)

AMPsequill is a computational antimicrobial peptide design submission. An ESM-C 300M model with a LoRA adapter generates a library of 50,000 peptides. XAMP, HemoPI2, and APEX predictions support a deterministic four-scenario ranking of 100 candidates. The submitted files are [`generate/library.fasta`](generate/library.fasta) and [`generate/top.fasta`](generate/top.fasta), in ranked order.

## Run the submission

Install [uv](https://docs.astral.sh/uv/) and Git LFS. The released outputs were generated on one NVIDIA V100 with FP16 and SDPA math attention. From the repository root:

```bash
git lfs install
git lfs pull
uv sync --frozen
uv run generate
```

The default command writes `generate/library.fasta` and `generate/top.fasta`. It uses seed 42, batch size 64, eight random-position unmasking steps, top-p 0.98, and temperatures 1.2→0.7. The adapter is the validation-selected optimizer step 175 from training with learning rate 5e-5, LoRA rank 16, and alpha 32. The ESM-C base is fetched from `biohub/esmc-300m-2024-12` at revision `7f10b20ae75017b2dbc884070e03434515709a8d`. Set `ESMC_WEIGHTS` to the pinned base-weight file for offline execution.

The generated library must match the released library before selection. The selector checks every sequence, accepted index, and sample ID against [`data/selection_scores.jsonl.gz`](data/selection_scores.jsonl.gz) and recomputes the ranking. This cache contains scores from the completed XAMP, APEX, HemoPI2, and public-reference screening run. HemoPI2 is replayed from the cache; its separate runtime and licensed assets are not bundled. A different library requires new scoring and validation. The program checks input and output integrity with SHA256; the submitted FASTA digests are in [`generate/submission-manifest.json`](generate/submission-manifest.json).

`--device auto` selects CUDA when available. CPU runs require an output directory outside `generate/` and are intended for diagnostics; the released files correspond to the V100 configuration. The default parameters are fixed at 50,000 sequences, 100 selected candidates, seed 42, and batch size 64.

## Candidate selection

Candidates must use the 20 standard amino acids, be 8–40 residues long, have no exact match to the included public competition reference, and have a maximum Levenshtein ratio of at most 0.80 to that reference. They must have XAMP integrated probability ≥0.5, HemoPI2 1.3 RF+MERCI hemolytic classifier score <0.46, and 11 positive, finite APEX predicted MIC values. These gates leave 5,594 candidates.

For each predicted MIC threshold of 8, 16, 32, and 64 µM, candidates are ranked by: seven-species predicted coverage (descending); coverage across the 11 endpoints (descending); the 90th percentile and median predicted MIC (ascending); and accepted index (ascending). Candidates are then ordered by their worst rank across the four scenarios, sum of scenario ranks, and accepted index. A greedy pass takes the first 100 with pairwise `Levenshtein.ratio` <0.65. No peptide was manually selected. The implementation is in [`selection.py`](src/ampsequill_amp_challenge_2027/selection.py).

The four thresholds are sensitivity scenarios for predicted MIC, not measured activity cutoffs. HemoPI2 supplies a classifier score, not HC50; the APEX predicted 90th percentile is not the competition's laboratory MIC90. These scores were used to select candidates, so they are not an independent assessment of efficacy or safety.

## Data and verification

[`data/TRAINING_DATA.md`](data/TRAINING_DATA.md) documents the public DBAASP, DRAMP, and dbAMP training sources, APD annotation source, filtering, cluster split, and sampling. The adapter was trained on 22,262 sequences with 2,474 cluster-held-out validation sequences. No non-public training set was used. Raw third-party database exports are not redistributed; derived length counts and a training-sequence membership guard needed by generation are included.

The library contains 50,000 unique sequences of length 8–50. The ranked 100 are a subset of it. All sequences use only standard amino acids and have no exact match to the included public antibacterial reference; the ranked list also has no similarity above 0.80 to that reference. Run `uv run --frozen python scripts/verify_frozen.py` to check the committed FASTA files, ranking replay, and integrity values without running the generator. The organizers' final reference set may differ from the public file. The private part of the APEX training inventory was unavailable for a strict leakage audit.

No wet-lab activity, hemolysis, official competition score, or clinical suitability is claimed. See [`NOTICE.md`](NOTICE.md) for third-party terms. Repository code is MIT-licensed; upstream model weights and tools retain their own terms. AI assistants supported software and documentation under human review.
