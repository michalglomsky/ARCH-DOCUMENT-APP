# ARCH-DOCUMENT-APP

Automated extraction of structured data from Polish construction permit application forms (wnioski o warunki zabudowy / pozwolenie na budowę) using a locally-run Vision Language Model (Qwen2.5-VL-7B) fine-tuned with LoRA on Apple Silicon (M3 Ultra).

---

## What it does

Given a scanned PDF of a construction permit application, the model extracts:

- Permit number, fill method, investment name, address, plot description
- Built-up area, facade width, floor counts, building height, roof geometry — per building
- Utilities (water, electricity, gas, heating, sewage)

Output is a structured JSON that maps directly to the master Excel spreadsheet (`Project Files/1-2026-DANE.xlsx`).

---

## Two parallel pipelines

This repo contains two independent fine-tuning approaches. Both use the same base model and the same source PDFs — the difference is in how training data is structured.

### `pipeline/page-level` — Original approach
Guide: `PIPELINE_GUIDE.md`

- Processes one **page image at a time**
- Extracts a flat JSON per page, merges results across pages
- Labels must be created manually (or bootstrapped via Nanonets)
- Good starting point; simpler data pipeline

### `pipeline/qa-pairs` — QA-pair approach (recommended)
Guide: `QA_PIPELINE_GUIDE.md`

- Processes all pages of a document **in a single model call**
- Uses `1-2026-DANE.xlsx` directly as ground truth — no manual labeling needed
- Outputs nested JSON with a `budynki` list (one entry per building) and a `media` list
- Better cross-page context; handles multi-building forms natively

| | Page-level | QA-pairs |
|---|---|---|
| Training unit | Single page | Full document |
| Labels | Manual / Nanonets | Already in Excel |
| Multi-building forms | Merge heuristic | Native list |
| Cross-page context | None | Full document |
| Ready to train immediately | No (labeling required) | Yes |

---

## Repository structure

```
ARCH-DOCUMENT-APP/
├── Project Files/
│   ├── 1-2026-DANE.xlsx          ← master label spreadsheet (tracked in git)
│   └── wz_*.pdf                  ← source permit PDFs (not tracked, too large)
├── finetune_qwen_vl_pytorch/     ← page-level pipeline code
│   ├── scripts/
│   ├── train/
│   └── requirements.txt
├── finetune_qwen_vl_qa/          ← QA-pair pipeline code
│   ├── scripts/
│   ├── train/
│   └── requirements.txt
├── demo_app/                     ← batch PDF → Excel runner + Nanonets comparison
├── PIPELINE_GUIDE.md             ← step-by-step for page-level pipeline
├── QA_PIPELINE_GUIDE.md          ← step-by-step for QA-pair pipeline
└── README.md                     ← this file
```

---

## Prerequisites

- macOS with Apple Silicon (MPS) — tested on M3 Ultra 96 GB
- Python 3.11
- PDFs in `Project Files/` (not committed to git)

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv finetune_qwen_vl_pytorch/.venv311
source finetune_qwen_vl_pytorch/.venv311/bin/activate
pip install -r finetune_qwen_vl_pytorch/requirements.txt
pip install -U "git+https://github.com/huggingface/transformers"
pip install openpyxl  # required for QA-pair pipeline only
```

---

## Where to start

- **To understand the full workflow:** read `QA_PIPELINE_GUIDE.md` — it is the more complete and easier-to-run pipeline since labels are already available in the Excel file.
- **To compare approaches:** run both pipelines to the evaluation step and open the resulting `.xlsx` files side by side.
- **To run inference on new PDFs:** start the server from either pipeline and POST a PDF path to `/extract`.

---

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Shared files: guides, `.gitignore`, README |
| `pipeline/page-level` | Page-level pipeline code |
| `pipeline/qa-pairs` | QA-pair pipeline code |
| `demo_1_ocr` | Original repo state (backup) |

When a pipeline is ready for production, merge its branch into `main` via a pull request.
