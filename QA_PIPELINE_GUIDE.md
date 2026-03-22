# QA-Pair Pipeline Guide — Qwen2.5-VL Document Extraction

End-to-end walkthrough using a **Question-Answer fine-tuning** approach:
the full document (all pages as images) is the **question**, and the
structured Excel record is the **answer**.

Compare results against the original page-by-page pipeline in `PIPELINE_GUIDE.md`.

---

## Why QA-pair tuning is a better fit here

| | Original pipeline | This pipeline |
|---|---|---|
| **Training unit** | Single page image | Full document (all pages) |
| **Labels** | Must be created manually per-page | Already exist in `1-2026-DANE.xlsx` |
| **Multi-building forms** | Flat dict, merge heuristics | Native `"budynki": [...]` list |
| **Cross-page context** | Lost between pages | Preserved — all pages in one call |
| **Label effort before first training** | High (need to label 200+ pages) | Near zero (Excel IS the labels) |

The Excel file has ~500 labeled records.  You can start training immediately
after parsing it — no manual labeling step required.

---

## Directory layout

```
arch-document-app/
├── finetune_qwen_vl_qa/
│   ├── scripts/
│   │   ├── parse_excel_labels.py   ← Excel → labels.json
│   │   ├── prepare_qa_dataset.py   ← PDFs + labels.json → train/val JSONL
│   │   ├── qa_utils.py             ← shared multi-image prompt/inference/scoring
│   │   ├── evaluate_qa.py          ← batch evaluation + Excel export
│   │   └── serve_vlm_qa.py         ← FastAPI server (port 8081)
│   ├── train/
│   │   └── train_lora_qa.py        ← LoRA fine-tuning (multi-image QA)
│   ├── data/                       ← generated images + JSONL live here
│   └── requirements.txt
└── QA_PIPELINE_GUIDE.md            ← this file
```

The QA pipeline **reuses the same `.venv311`** as the original pipeline —
no second environment needed.  Only one new dependency is added: `openpyxl`
(for reading/writing Excel files).

```bash
source finetune_qwen_vl_pytorch/.venv311/bin/activate
pip install openpyxl   # one-time addition
```

---

## Step 0 — Verify the environment

```bash
source finetune_qwen_vl_pytorch/.venv311/bin/activate
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Expected: MPS: True
python -c "import openpyxl; print('openpyxl OK')"
```

---

## Step 1 — Parse the Excel labels

This converts `1-2026-DANE.xlsx` into a `labels.json` file that maps each
permit number to a structured Python dict.

```bash
source finetune_qwen_vl_pytorch/.venv311/bin/activate

python finetune_qwen_vl_qa/scripts/parse_excel_labels.py \
    --excel   "/Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup/1-2026-DANE.xlsx" \
    --output  finetune_qwen_vl_qa/data/labels.json
```

Expected output:
```
Parsed 487 permit records → finetune_qwen_vl_qa/data/labels.json
  with building data:    479
  multi-building forms:  312
```

Open `finetune_qwen_vl_qa/data/labels.json` to verify a few records look right.
Each record has this shape:

```json
{
  "808": {
    "nr_wniosku": "808",
    "sposob_wypelnienia": "Komputerowo",
    "flaga_7_9": "BRAK",
    "nazwa_inwestycji": "BUDOWA BUDYNKU MIESZKALNEGO I BUDYNKU GOSPODARCZEGO ...",
    "adres": "Łąki 97, 08-404 Górzno",
    "teren_inwestycji": "teren części działki o numerze ewidencyjnym 1142/11 ...",
    "pow_zabudowy_calosc": "od 90,0 m2 do 350,0 m2",
    "budynki": [
      {
        "oznaczenie": "1. Mieszkalny",
        "szerokosc_elewacji": "od 8,0 do 25,0 m",
        "suma_pow_nadziemnych": "od 90,0 do 280,0 m2",
        "suma_pow_podziemnych": "nie dotyczy",
        "wys_gornej_krawedzi": "od 2,5 do 9,0 m",
        "wysokosc_zabudowy": "od 5,0 do 9,0 m",
        "ilosc_kond_nadziemnych": "max 2",
        "ilosc_kond_podziemnych": "1",
        "geometria_dachu": "1/2/4/wielo (15-45°)"
      },
      {
        "oznaczenie": "2. Gospodarczy",
        ...
      }
    ],
    "media": ["Woda: studnia", "Prąd: z sieci", "Ciepło: indyw. kotłownia", "Ścieki: indyw. oczyszczalnia"]
  }
}
```

---

## Step 2 — Build the QA dataset

This renders PDFs to images and pairs each document with its Excel label.

```bash
python finetune_qwen_vl_qa/scripts/prepare_qa_dataset.py \
    --pdf-dir     "/Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup" \
    --labels-json finetune_qwen_vl_qa/data/labels.json \
    --output-dir  finetune_qwen_vl_qa/data \
    --dpi         200 \
    --max-pages   6 \
    --val-split   0.15
```

### Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--dpi` | 200 | Lower than original (250) because we feed 4-6 images at once |
| `--max-pages` | 6 | Pages 1-4 typically contain all key fields for WZ forms |
| `--val-split` | 0.15 | Split is by document (no leakage) |

### Output

```
finetune_qwen_vl_qa/data/
├── images/
│   ├── wz_808_p001.png
│   ├── wz_808_p002.png
│   └── ...
├── train.jsonl       ← one record per DOCUMENT (not per page)
├── val.jsonl
└── unlabeled.jsonl   ← PDFs with no matching Excel row (wz_un_*, wz_zal_*, ...)
```

Each JSONL record:
```json
{
  "pdf_stem":   "wz_808",
  "nr_wniosku": "808",
  "image_paths": [
    "/abs/path/data/images/wz_808_p001.png",
    "/abs/path/data/images/wz_808_p002.png",
    ...
  ],
  "prompt":     "You are a document extraction assistant ...",
  "target_json": { ... }
}
```

### Expected match rate

Not every PDF in the backup folder has a row in the Excel — some are
`wz_un_*` (unknown series) or `wz_zal_*` (attachments). The script reports
how many matched and lists the unmatched ones in `unlabeled.jsonl`.

---

## Step 3 — Baseline evaluation (zero-shot)

Run before training to measure how much the model already knows.

```bash
python finetune_qwen_vl_qa/scripts/evaluate_qa.py \
    --val-jsonl   finetune_qwen_vl_qa/data/val.jsonl \
    --model       Qwen/Qwen2.5-VL-7B-Instruct \
    --output      eval_qa_baseline.json \
    --output-xlsx eval_qa_baseline.xlsx
```

Expected output example:
```
Overall field accuracy: 44.2%

Per-field accuracy:
  nr_wniosku                        81.3%  ████████████████
  adres                             62.1%  ████████████
  bld1_geometria_dachu              38.4%  ███████
  media                             29.7%  █████
  ...
```

Open `eval_qa_baseline.xlsx` alongside the backup `1-2026-DANE.xlsx` to
compare column by column.

---

## Step 4 — LoRA fine-tuning

```bash
python finetune_qwen_vl_qa/train/train_lora_qa.py \
    --model       Qwen/Qwen2.5-VL-7B-Instruct \
    --train-jsonl finetune_qwen_vl_qa/data/train.jsonl \
    --val-jsonl   finetune_qwen_vl_qa/data/val.jsonl \
    --out-dir     finetune_qwen_vl_qa/out/lora_run1 \
    --epochs      2 \
    --lr          5e-5 \
    --grad-accum  8 \
    --lora-r      16 \
    --lora-alpha  32 \
    --save-every  50
```

### What to watch

| Signal | Good | Warning |
|--------|------|---------|
| Train loss | Decreasing steadily | Stuck or NaN → lower LR |
| Val loss | Decreasing or flat | Rising → overfitting (reduce epochs) |
| Loss at step 0 | < 5.0 | Very high → check label masking |
| Skipped samples | < 10% | Many skipped → lower `--max-pages` or `--max-pixels` |

### Key flags vs. original pipeline

| Flag | Original | QA pipeline | Why different |
|------|----------|-------------|---------------|
| `--grad-accum` | 16 | **8** | Each sample is a full document (~5× more tokens than one page) |
| `--max-seq-len` | 2048 | **6144** | Multi-image sequences are longer |
| `--max-pixels` | 1_003_520 | **501_760** | Halved per image to fit 6 images in memory |
| `--save-every` | 100 | **50** | Fewer optimizer steps total (document-level, not page-level) |

### Memory usage on M3 Ultra (96 GB)

- Qwen2.5-VL-7B float16: ~14 GB
- 6 page images × ~512 tokens each: +6 GB activations
- LoRA adapters + optimizer: ~20 GB
- Total estimate: ~40–50 GB (within 96 GB)

If you hit OOM:
- Add `--max-pixels 250880` (reduces to ~256 tokens/image)
- Or reduce `--max-pages 4`

### Output

```
finetune_qwen_vl_qa/out/lora_run1/
├── adapter_config.json
├── adapter_model.safetensors
└── ...processor files...
```

---

## Step 5 — Post-training evaluation

```bash
python finetune_qwen_vl_qa/scripts/evaluate_qa.py \
    --val-jsonl      finetune_qwen_vl_qa/data/val.jsonl \
    --model          Qwen/Qwen2.5-VL-7B-Instruct \
    --lora-adapter   finetune_qwen_vl_qa/out/lora_run1 \
    --output         eval_qa_finetuned.json \
    --output-xlsx    eval_qa_finetuned.xlsx
```

Compare `eval_qa_baseline.json` vs `eval_qa_finetuned.json` to measure gains.
Open both `.xlsx` files side-by-side to visually review extraction quality.

---

## Step 6 — Start the inference server

```bash
# Zero-shot:
python finetune_qwen_vl_qa/scripts/serve_vlm_qa.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8081

# Fine-tuned:
python finetune_qwen_vl_qa/scripts/serve_vlm_qa.py \
    --model        Qwen/Qwen2.5-VL-7B-Instruct \
    --lora-adapter finetune_qwen_vl_qa/out/lora_run1 \
    --port 8081
```

Note: the original pipeline runs on port 8080; the QA server uses **8081**
so both can run simultaneously for comparison.

Wait for: `Model ready on mps.`

### Health check

```bash
curl http://127.0.0.1:8081/health
# {"status":"ok","device":"mps","model":"Qwen/...","adapter":"..."}
```

### Extract a single PDF

```bash
curl -s -X POST http://127.0.0.1:8081/extract \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "/abs/path/to/wz_808.pdf", "max_pages": 6}' \
  | python3 -m json.tool
```

Expected response:
```json
{
  "nr_wniosku": "808",
  "sposob_wypelnienia": "Komputerowo",
  "flaga_7_9": "BRAK",
  "nazwa_inwestycji": "BUDOWA BUDYNKU MIESZKALNEGO ...",
  "adres": "Łąki 97, 08-404 Górzno",
  "teren_inwestycji": "teren części działki ...",
  "pow_zabudowy_calosc": "od 90,0 m2 do 350,0 m2",
  "budynki": [ {...}, {...} ],
  "media": ["Woda: studnia", "Prąd: z sieci", ...],
  "needs_review": false
}
```

### Batch extract an entire folder to Excel

```bash
curl -s -X POST http://127.0.0.1:8081/extract_to_xlsx \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_dir": "/Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup",
    "output_xlsx": "/abs/path/results_qa.xlsx"
  }'
```

---

## Comparing QA pipeline vs. original pipeline

| Metric | Original (page-level) | QA (document-level) |
|--------|----------------------|---------------------|
| Labeled samples available | Must label manually | ~500 from Excel immediately |
| Handles multi-building | Via merge heuristic | Native list field |
| Cross-page context | None | Full document |
| Tokens per training step | ~600–1200 | ~3000–5000 |
| Training time per epoch | Faster (more, shorter steps) | Slower (fewer, longer steps) |
| `needs_review` quality | Per-page estimate | Per-document estimate |

Run both servers (`--port 8080` and `--port 8081`) and send the same PDFs to
each to build a direct comparison table.

---

## Iteration loop (how to improve)

```
Review eval_qa_finetuned.xlsx  →  Find failure patterns  →  Fix labels  →  Retrain
```

1. Open `eval_qa_finetuned.xlsx` — low `Field accuracy` rows indicate hard cases.
2. Common failure types:
   - Wrong `nr_wniosku` → form number on an unusual position → add similar examples
   - Missing building → two buildings on one form not split correctly → fix `labels.json` and retrain
   - Wrong media → handwritten media section → add more handwritten examples
3. Fix labels directly in `labels.json` (not in the Excel) and re-run Steps 2 & 4.

---

## Troubleshooting

### `processor error for wz_XXX` during training
Too many images.  Reduce `--max-pages` to 4 or `--max-pixels` to `250880`.

### Training loss is NaN
- Lower `--lr` to `1e-5`
- Verify `labels.json` has valid JSON in all `target_json` values:
  ```bash
  python3 -c "
  import json; d = json.load(open('finetune_qwen_vl_qa/data/labels.json'))
  for k,v in d.items():
      try: json.dumps(v)
      except Exception as e: print(k, e)
  "
  ```

### Many samples skipped (too long)
- Lower `--max-pages 4` (most WZ forms have key fields on pages 1-4)
- Or lower `--max-pixels 250880`

### `KeyError: 'qwen2_5_vl'`
Install transformers from source:
```bash
pip install -U "git+https://github.com/huggingface/transformers"
```

### Excel has garbled Polish characters
`openpyxl` handles UTF-8 natively.  If you see `?` characters, check that
you opened the file in Excel with UTF-8 encoding (File → Import or use
LibreOffice Calc).

---

## Notes on MPS vs. CUDA

Same as the original pipeline: Unsloth requires CUDA and does not work on MPS.
These scripts use plain HuggingFace PEFT + PyTorch and are correct for your
M3 Ultra.  The main practical difference from the original pipeline is that
multi-image sequences are substantially longer, so `max_seq_len` must be
higher and `grad_accum` lower.
