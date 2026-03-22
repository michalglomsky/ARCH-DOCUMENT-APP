## Local fine-tuning environment (PyTorch/MPS) — Qwen2.5‑VL

This folder sets up a **local** Python environment for experimenting with **LoRA/QLoRA-style** fine-tuning using **PyTorch on Apple Silicon (MPS)**.

### Important note (Python version for Qwen2.5‑VL)

Your system `python3` is **3.9.6**, but Qwen2.5‑VL requires a Transformers build that (as of now) needs **Python ≥ 3.10**.
This setup includes a **Python 3.11 venv** path that works on macOS Apple Silicon.

---

## 1) Create the virtual environment and install dependencies

From the repo root (`OpenClawConfig/`):

```bash
# Install Python 3.11 (once)
brew install python@3.11

# Create venv (recommended)
/opt/homebrew/bin/python3.11 -m venv finetune_qwen_vl_pytorch/.venv311
source finetune_qwen_vl_pytorch/.venv311/bin/activate

python -m pip install --upgrade pip
pip install -r finetune_qwen_vl_pytorch/requirements.txt
```

### Install a Transformers build that supports Qwen2.5‑VL

Per the model card for `[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)`, install Transformers from source to avoid `KeyError: 'qwen2_5_vl'`:

```bash
pip install -U "git+https://github.com/huggingface/transformers" accelerate
```

Quick sanity checks:

```bash
python -c "import torch; print('torch', torch.__version__); print('mps', torch.backends.mps.is_available())"
python -c "import transformers, peft, accelerate; print(transformers.__version__, peft.__version__, accelerate.__version__)"
```

---

## 2) Next steps (training entrypoint)

This env is the prerequisite. The actual training script depends on:

- which exact Qwen2.5‑VL checkpoint you’ll use (HF vs converted)
- your dataset format (image paths + JSON targets)
- whether you fine-tune the full VLM or only adapters

Once you confirm the exact model ID you want to fine-tune (7B vs 32B and the specific checkpoint name), I can add:

- a concrete `train_lora.py` training entrypoint
- a dataset prep script (`pdf -> images` + JSONL)
- a minimal local inference server endpoint that plugs into `demo_app/` (`local_vlm.endpoint`)

