# Qwen3 Turkish LoRA Kaggle pilot

This public repository contains the training code and Kaggle notebook only.
The reviewed conversation JSONL is project-authored, AI-assisted material
that is not approved for public redistribution and must not be committed here.

## Kaggle setup

1. Create a **private** Kaggle Dataset containing
   `qwen3_dialogue_review_001.jsonl` from the project's approved data file.
2. Create a Kaggle notebook by importing
   [`kaggle_qwen3_0_6b_lora.ipynb`](./kaggle_qwen3_0_6b_lora.ipynb).
3. The notebook is configured to clone this public repository. Attach the
   private dataset, enable a GPU accelerator and internet access, then run all
   cells.
4. Save the Kaggle output directory containing the LoRA adapter, training
   metadata, and one-shot held-out comparison.

The notebook validates the private JSONL SHA-256 and canonical dataset
fingerprint before training. The 20 conversations are split by group into
16 train, 2 validation, and 2 held-out test conversations. The test comparison
is intentionally one-shot; this small pilot does not establish general
conversational quality.

Assistant-only loss masking uses character offsets from the rendered chat
template and tokenizes both training and generation text through the same
tokenizer path. It does not assume chat-template token IDs exactly match
offset-producing tokenization.

## Local checks

Install the dependencies from `requirements.txt`, then run:

```powershell
python -m unittest discover -s tests -v
```

The local test suite checks assistant-only loss masking without requiring the
private dataset or a GPU.