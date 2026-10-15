# Qwen3 Turkish LoRA experiments

This repository contains the Qwen3-0.6B LoRA trainer and two Kaggle
experiments:

- [`kaggle_qwen3_0_6b_lora.ipynb`](./kaggle_qwen3_0_6b_lora.ipynb) trains
  with 20 project-authored examples. Their JSONL remains private and is not
  included in this repository. Use the original notebook's private Kaggle
  Dataset workflow only if you have that approved file.
- [`kaggle_qwen3_0_6b_lora_expanded.ipynb`](./kaggle_qwen3_0_6b_lora_expanded.ipynb)
  trains with the expanded, licensed dialogue derivative committed in
  [`datasets/qwen3_0_6b_lora_bundle_002.zip`](./datasets/qwen3_0_6b_lora_bundle_002.zip)
  and optionally combines the 20 owner-approved project examples from a
  **separate private Kaggle Dataset**. The public data and code download from
  the pinned GitHub tag automatically; private data is never stored here.

## Run the expanded experiment

1. Upload `artifacts/qwen3_private_dialogue_addon_001.zip` from the local
   project as a **private** Kaggle Dataset, and attach it to the notebook.
   This one-time upload contains the approved 20 project dialogues. They are
   not approved for public redistribution and must not be added to GitHub.
2. Open the new combined-data notebook in Kaggle:
   [pinned GitHub notebook](https://github.com/dozagoat47-star/qwen3-turkish-lora-kaggle/blob/qwen3-combined-private-001/kaggle_qwen3_0_6b_lora_expanded.ipynb).
3. Enable a GPU accelerator and internet access.
4. Run all cells. The notebook fetches the public licensed bundle from GitHub,
   discovers the private Dataset by its fixed ZIP filename, checks both
   datasets' hashes, provenance, and review status, and verifies all 20
   project examples remain in the train split.
5. Download the single output file
   `qwen3_0_6b_lora_run_003.zip` from the Kaggle notebook's output files.

The expanded dataset is the train-only derivative of Turkish Daily Dialogues
5K, licensed CC BY 4.0. Its exact attribution and license are included in the
bundle. The conversations are synthetic and remain **draft**; they were not
individually human-reviewed. The experiment's draft-data opt-in is explicit
in the notebook and trainer. Scenario families stay intact across
train/validation/test splits. Validation loss selects the adapter; a
deterministic sample of at most 32 held-out assistant turns is compared once.
This pilot does not establish general Turkish quality or factual reliability.

The notebook converts the resulting LoRA weights into a llama.cpp GGUF
adapter. Keep `adapter_model.gguf` together with the existing Q4_K_M base
model; the adapter alone is not a complete model.

The setup removes Kaggle's incompatible optional `torchao` package without
changing Torch/CUDA. The final export cell verifies the pinned llama.cpp
revision, confirms that `adapter_model.gguf` was created, records its hash,
and only then packages the outputs. LoRA training runs on the selected
Kaggle GPU, while local inference uses the existing llama.cpp runtime.

## Local checks

Install the dependencies from `requirements.txt`, then run:

```powershell
python -m unittest discover -s tests -v
```

The tests cover assistant-only loss masking, group-safe splitting, and
training-data validation without requiring Kaggle or a GPU.