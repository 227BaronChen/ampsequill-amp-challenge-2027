# AMPsequill LoRA adapter

This directory contains the validation-selected adapter used by the AMP Challenge 2027 generator.

- Base model: `biohub/esmc-300m-2024-12`, revision `7f10b20ae75017b2dbc884070e03434515709a8d`
- Training data: public DBAASP, DRAMP, and dbAMP sequences described in [`data/TRAINING_DATA.md`](../../data/TRAINING_DATA.md)
- Initial learning rate: `5e-5`; best optimizer step: `175`
- LoRA rank / alpha / dropout: `16 / 32 / 0`
- Target modules: ESM-C attention `layernorm_qkv.1` and `out_proj`
- Training seed: `42`

The generator checks the adapter's file integrity before use. Predictions are not validated MIC, toxicity, or clinical-use guarantees. The base model's terms remain applicable; see [`NOTICE.md`](../../NOTICE.md).
