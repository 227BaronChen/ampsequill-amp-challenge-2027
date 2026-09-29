# Training-data disclosure

The AMPsequill adapter was trained on 22,262 public peptide sequences, with 2,474 sequences held out for validation by sequence-cluster connected components. The preprocessing pipeline parsed each source, uppercased sequences, kept the 20 standard amino acids and lengths 8–50, required supported antibacterial annotations, removed exact duplicates and incompatible peptide chemistry, then split clusters with MMseqs2. Sampling weighted each cluster by `1/sqrt(cluster_size)` with seed 42.

The ESM-C adapter used LoRA rank 16, alpha 32, dropout 0, micro-batch 64, gradient accumulation 4, initial learning rate 5e-5, and the validation-selected checkpoint at optimizer step 175.

## Public sources

| Source | Role | Version / accessed | Link | Records after per-source sequence normalization |
|---|---|---|---|---:|
| DBAASP activity/API | Antibacterial MIC records | Snapshot accessed 2026-09-09 | [DBAASP API](https://dbaasp.org/peptides?format=json) | 10,826 |
| DRAMP antibacterial export | Antibacterial sequence annotations | DRAMP 3.0, accessed 2026-09-09 | [DRAMP export](https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antibacterial_amps.fasta) | 24,046 |
| dbAMP antibacterial export | Antibacterial sequence annotations | dbAMP 2.0 (2021-06), accessed 2026-09-09 | [dbAMP export](https://ycclab.cuhk.edu.cn/dbAMP/download/2.0/activity/dbAMP_Antibacterial.tar.gz) | 5,052 |
| APD natural AMP export | Provenance and overlap annotation only | APD 2024a, accessed 2026-09-09 | [APD export](https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta) | 3,306 |

These counts precede cross-source exact deduplication. APD was used for annotation and overlap checks, not as an additional training-label source. No private or unpublished training data was used. Raw third-party database exports are not redistributed. The repository includes the derived training-length distribution and exact-membership SHA256 values used by the generator to reject training-sequence matches.

Sequences with noncanonical residues, lengths outside 8–50, duplicate exact sequences, incompatible modification or terminus annotations, or unsupported antibacterial labels were excluded. The public competition reference was used as a generation-time exclusion and selection reference, not as a validation set.
