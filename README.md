# Qwen3 Turkish LoRA experiments

This repository contains the Qwen3-0.6B LoRA trainer and two Kaggle
experiments:

- [`kaggle_qwen3_0_6b_lora.ipynb`](./kaggle_qwen3_0_6b_lora.ipynb) trains
  with 20 project-authored examples. Their JSONL remains private and is not
  included in this repository. Use the original notebook's private Kaggle
  Dataset workflow only if you have that approved file.
- [`kaggle_qwen3_0_6b_lora_expanded.ipynb`](./kaggle_qwen3_0_6b_lora_expanded.ipynb)
  trains with the expanded, licensed dialogue derivative committed in
  [`datasets/qwen3_0_6b_lora_bundle_002.zip`](./datasets/qwen3_0_6b_lora_bundle_002.zip).
  It downloads and verifies its own pinned GitHub inputs; there is no Kaggle
  Dataset upload or file selection step.

## Run the expanded experiment

1. Open the expanded notebook in Kaggle using
   [this GitHub notebook](https://github.com/dozagoat47-star/qwen3-turkish-lora-kaggle/blob/qwen3-expanded-draft-001/kaggle_qwen3_0_6b_lora_expanded.ipynb).
2. Enable a GPU accelerator and internet access.
3. Run all cells. The notebook checks the bundle hashes, dataset provenance,
   review status, and train-only source before training.
4. Download the single output file
   `qwen3_0_6b_lora_run_002.zip` from the Kaggle notebook's output files.

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
changing Torch/CUDA. LoRA training runs on the selected Kaggle GPU, while
local inference uses the existing llama.cpp runtime.

## Local checks

Install the dependencies from `requirements.txt`, then run:

```powershell
python -m unittest discover -s tests -v
```

The tests cover assistant-only loss masking, group-safe splitting, and
training-data validation without requiring Kaggle or a GPU.