# Pipeline Guide — Qwen2.5-VL Document Extraction

End-to-end walkthrough: from raw PDFs to a fine-tuned local VLM that extracts
construction permit form fields, with a side-by-side comparison against the
Nanonets API.

---

## Overview of what was built

```
OpenClawConfig/
├── finetune_qwen_vl_pytorch/
│   ├── scripts/
│   │   ├── vlm_utils.py          ← shared prompt/inference/parsing utilities
│   │   ├── prepare_dataset.py    ← PDFs → page images + train/val JSONL
│   │   ├── serve_local_vlm.py    ← local FastAPI inference server
│   │   └── evaluate.py           ← batch evaluation + metrics
│   ├── train/
│   │   └── train_lora.py         ← LoRA fine-tuning (MPS-correct label masking)
│   ├── data/                     ← generated images + JSONL live here
│   └── requirements.txt
└── demo_app/
    ├── extractor_demo/
    │   ├── cli.py                ← batch PDF → XLSX runner
    │   ├── providers.py          ← Nanonets | LocalVLM provider switch
    │   └── config.py
    └── config.example.yaml       ← schema fields + provider config
```

---

## Prerequisites

### 1. Python 3.11 environment

```bash
brew install python@3.11

/opt/homebrew/bin/python3.11 -m venv finetune_qwen_vl_pytorch/.venv311
source finetune_qwen_vl_pytorch/.venv311/bin/activate

python -m pip install --upgrade pip
pip install -r finetune_qwen_vl_pytorch/requirements.txt

# Qwen2.5-VL requires the latest transformers (not yet on PyPI)
pip install -U "git+https://github.com/huggingface/transformers" accelerate
```

Verify MPS is available:
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Expected: MPS: True
```

### 2. demo_app environment (separate, lighter)

```bash
cd demo_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

---

## Step 1 — Render PDFs to images + create dataset JSONL

This converts your PDF folder into per-page PNG images and scaffold JSONL files.

```bash
source finetune_qwen_vl_pytorch/.venv311/bin/activate

python finetune_qwen_vl_pytorch/scripts/prepare_dataset.py \
    --pdf-dir  "Project Files" \
    --output-dir finetune_qwen_vl_pytorch/data \
    --dpi 250 \
    --val-split 0.15
```

Output:
```
finetune_qwen_vl_pytorch/data/
├── images/
│   ├── 945_p001.png
│   ├── 945_p002.png
│   └── ...
├── train.jsonl
└── val.jsonl
```

Each JSONL record looks like:
```json
{
  "image_path": "/abs/path/finetune_qwen_vl_pytorch/data/images/945_p001.png",
  "prompt": "You are a document extraction assistant...",
  "target_json": {}   ← fill this in (Step 2)
}
```

---

## Step 2 — Label your data (fill in target_json)

You need ground-truth labels before training. Three options:

### Option A — Nanonets API (fastest for bulk labeling)

1. Upload PDFs to Nanonets and run your extraction workflow.
2. Export the results as a JSONL file where each line has:
   ```json
   {"image_path": "945_p001.png", "target_json": {"nr_wniosku": "945", ...}}
   ```
3. Re-run prepare_dataset.py with `--labels-jsonl` to merge:
   ```bash
   python finetune_qwen_vl_pytorch/scripts/prepare_dataset.py \
       --pdf-dir "Project Files" \
       --output-dir finetune_qwen_vl_pytorch/data \
       --labels-jsonl nanonets_export.jsonl
   ```
4. Manually correct any Nanonets errors (these become high-quality training signal).

### Option B — Zero-shot bootstrap (use model to pre-fill, then correct)

See Step 3 (Baseline). After running evaluate.py zero-shot, the output
`eval_baseline.json` contains the model's predictions. Copy them to
`target_json`, then correct the mistakes manually.

### Option C — Manual labeling

Open `train.jsonl` in any text editor or write a small script to display each
image and prompt for field values. Tedious but highest quality.

**Minimum viable labeled set:**
- 200–500 page images → first iteration (fast experiment)
- 1,000–3,000 page images → production-quality model

---

## Step 3 — Baseline evaluation (zero-shot, no fine-tuning)

This tells you how good the base model already is and gives you a comparison
point after fine-tuning.

Requires at least some labeled records in val.jsonl.

```bash
source finetune_qwen_vl_pytorch/.venv311/bin/activate

python finetune_qwen_vl_pytorch/scripts/evaluate.py \
    --val-jsonl finetune_qwen_vl_pytorch/data/val.jsonl \
    --model     Qwen/Qwen2.5-VL-7B-Instruct \
    --output    eval_baseline.json
```

Expected runtime: ~1–3 min to load model + ~30s per sample on MPS.

Example output:
```
Overall field accuracy:  52.3%
Per-field accuracy:
  nr_wniosku                      87.5%  █████████████████
  adres                           61.2%  ████████████
  pow_zabudowy_calosc             43.0%  ████████
  ...
needs_review  precision=0.71  recall=0.58
```

---

## Step 4 — LoRA fine-tuning

Run after you have labeled train.jsonl and val.jsonl.

```bash
source finetune_qwen_vl_pytorch/.venv311/bin/activate

python finetune_qwen_vl_pytorch/train/train_lora.py \
    --model       Qwen/Qwen2.5-VL-7B-Instruct \
    --train-jsonl finetune_qwen_vl_pytorch/data/train.jsonl \
    --val-jsonl   finetune_qwen_vl_pytorch/data/val.jsonl \
    --out-dir     finetune_qwen_vl_pytorch/out/lora_run1 \
    --epochs      2 \
    --lr          5e-5 \
    --grad-accum  16 \
    --lora-r      16 \
    --lora-alpha  32 \
    --save-every  100
```

### What to watch

| Signal | Good | Warning |
|--------|------|---------|
| Train loss | Decreasing steadily | Stuck or NaN → lower LR |
| Val loss | Decreasing or flat | Rising → overfitting (add dropout / reduce epochs) |
| Loss at step 0 | < 5.0 | Very high → check label masking |

### Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--lora-r` | 16 | Higher = more capacity, more memory |
| `--grad-accum` | 16 | Effective batch = grad_accum × 1 |
| `--max-seq-len` | 2048 | Samples longer than this are skipped |
| `--save-every` | 100 | Checkpoint every N optimizer steps |

### Memory usage on M3 Ultra (96 GB)

- Qwen2.5-VL-7B in float16: ~14 GB
- LoRA adapters + activations + optimizer: ~20–30 GB
- Total: ~35–45 GB (well within 96 GB)

### Output

```
finetune_qwen_vl_pytorch/out/lora_run1/
├── adapter_config.json
├── adapter_model.safetensors
└── ...processor files...
```

---

## Step 5 — Post-training evaluation

Compare the fine-tuned model against the baseline from Step 3.

```bash
python finetune_qwen_vl_pytorch/scripts/evaluate.py \
    --val-jsonl    finetune_qwen_vl_pytorch/data/val.jsonl \
    --model        Qwen/Qwen2.5-VL-7B-Instruct \
    --lora-adapter finetune_qwen_vl_pytorch/out/lora_run1 \
    --output       eval_finetuned.json
```

Compare `eval_baseline.json` vs `eval_finetuned.json` to measure gains.

---

## Step 6 — Start the local VLM server

The server loads the model once and serves the demo_app's `/extract` endpoint.

```bash
source finetune_qwen_vl_pytorch/.venv311/bin/activate

# Zero-shot:
python finetune_qwen_vl_pytorch/scripts/serve_local_vlm.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8080

# With fine-tuned adapter:
python finetune_qwen_vl_pytorch/scripts/serve_local_vlm.py \
    --model        Qwen/Qwen2.5-VL-7B-Instruct \
    --lora-adapter finetune_qwen_vl_pytorch/out/lora_run1 \
    --port 8080
```

Wait for: `Model ready on mps. Serving on http://127.0.0.1:8080`

Quick health check:
```bash
curl http://127.0.0.1:8080/health
# {"status":"ok","device":"mps"}
```

Manual test:
```bash
curl -s -X POST http://127.0.0.1:8080/extract \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "/abs/path/to/test.pdf", "schema_fields": ["nr_wniosku","adres","needs_review"]}' \
  | python3 -m json.tool
```

---

## Step 7 — Run demo_app (local VLM vs Nanonets comparison)

### Local VLM provider

```bash
source demo_app/.venv/bin/activate

# Edit demo_app/config.yaml:
#   provider:
#     type: "local_vlm"
#   local_vlm:
#     endpoint: "http://127.0.0.1:8080/extract"

python -m extractor_demo.cli \
    --input-dir  "Project Files" \
    --output-file results_local_vlm.xlsx \
    --config     demo_app/config.yaml
```

### Nanonets provider

```bash
export NANONETS_API_KEY="your-key-here"

# Edit demo_app/config.yaml:
#   provider:
#     type: "nanonets"
#   nanonets:
#     endpoint: "https://app.nanonets.com/api/v2/..."

python -m extractor_demo.cli \
    --input-dir  "Project Files" \
    --output-file results_nanonets.xlsx \
    --config     demo_app/config.yaml
```

Both outputs have identical columns → open in Excel side-by-side to compare.

---

## Iteration loop (how to improve the model)

```
Identify failures  →  Add hard examples to train set  →  Retrain  →  Evaluate
```

1. Look at `eval_finetuned.json` → find samples where `field_scores` has many `false`.
2. Open those images — categorise failure type:
   - Bad scan → add more similar scans
   - Unusual handwriting → find more handwriting examples
   - Multi-building entries (e.g. comma-joined values) → add explicit prompt instruction
3. Add corrected records to `train.jsonl`.
4. Re-run Steps 4–5.

---

## Troubleshooting

### `KeyError: 'qwen2_5_vl'` on model load
Install transformers from source:
```bash
pip install -U "git+https://github.com/huggingface/transformers"
```

### MPS out of memory
- Reduce `--max-seq-len` (e.g. `--max-seq-len 1536`)
- Lower DPI: `--dpi 200` in prepare_dataset.py and serve_local_vlm.py
- Reduce `--lora-r` to 8

### Training loss is NaN
- Lower LR: `--lr 1e-5`
- Check that val.jsonl has valid JSON in `target_json`

### Server returns `_parse_error`
The model output could not be parsed as JSON. Likely zero-shot on an unusual form.
Fine-tuning with examples of that form type will fix this.

### Nanonets endpoint 401/403
Check that `NANONETS_API_KEY` is set and that `nanonets.endpoint` in config.yaml
points to your correct workflow extraction URL.

---

## Important notes for MPS (vs CUDA/Unsloth)

The plan document (`VLM_OCR_COSTS_AND_PLAN.md`) recommends Unsloth — but
**Unsloth requires CUDA and does not support MPS**.

The scripts in this repo use plain HuggingFace PEFT + PyTorch, which is the
correct path for your M3 Ultra. The label masking logic is implemented from
scratch to be equivalent to Unsloth's `train_on_responses_only=True`.

Practical implications:
- Training is slower than on an A100 (no kernel optimisations) but your 96 GB
  RAM more than compensates for the 7B model.
- If you ever want to run on a cloud GPU, the same scripts work — just swap
  `torch_dtype=torch.bfloat16` and remove the MPS-specific `empty_cache()` calls.
