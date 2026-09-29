# Third-party notices

The repository code is MIT-licensed. Third-party assets retain their own
licenses:

- ESMC-300M base weights: Biohub/EvolutionaryScale license terms at
  <https://huggingface.co/biohub/esmc-300m-2024-12> and
  <https://github.com/Biohub/esm/blob/main/LICENSE.md>. The base weight is
  downloaded at the pinned revision and is not copied into this repository.
- APEX-pathogen inference code and checkpoints: the MIT license shipped in
  `third_party/apex/LICENSE`.
- ESM-2 (`facebook/esm2_t12_35M_UR50D`) and the Transformers implementation
  used by XAMP-E: retain the upstream license and citation supplied by the
  model and library. The inference code pins and checks the model assets.
- Source AMP databases are public upstream resources; this repository does
  not redistribute their raw records. See `data/TRAINING_DATA.md`.
