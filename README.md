# ARCH-DOCUMENT-APP

Automated extraction of structured data from Polish construction permit application forms (wnioski o warunki zabudowy / pozwolenie na budowę) using a locally-run Vision Language Model (Qwen2.5-VL-7B) fine-tuned with LoRA on Apple Silicon (M3 Ultra).

---

## Tools

### Extraction tool — `./start.sh`

Browser UI that sends scanned PDFs to a local VLM and extracts structured data into an Excel spreadsheet.

- Starts two servers: a **VLM server** (port 8081) loading Qwen2.5-VL and an **app server** (port 8000) serving the browser UI
- Extracts: permit number, fill method, investment name, address, plot description, built-up area, facade width, floor counts, building height, roof geometry, utilities
- Output maps directly to the master Excel spreadsheet

```bash
./start.sh                              # zero-shot model
./start.sh --lora-adapter path/to/out   # with LoRA fine-tuned adapter
./start.sh --port-app 8001              # custom port
./start.sh --max-pages 4                # limit pages per inference
```

Open `http://localhost:8000` in your browser.

---

### Manual redaction tool — `./run_manual_redaction.sh`

Browser UI for drawing redaction boxes over PDF pages and saving redacted copies.

- Renders PDF pages in the browser; click and drag to draw redaction zones
- Supports batch redaction across a whole directory of PDFs
- Redacted files are saved with a `.redacted.pdf` suffix

```bash
./run_manual_redaction.sh
```

Open `http://localhost:8083` in your browser. See `manual_redaction/MANUAL_REDACTION_GUIDE.md` for full usage.

---

## Fine-tuning pipeline

The QA-pair pipeline fine-tunes the model on your own labeled documents.

### 1. Build the dataset — `./run_preprocess.sh`

Renders PDF pages to images and builds `train.jsonl` / `val.jsonl` / `test.jsonl` from the master Excel file (`1-2026-DANE.xlsx`). No manual labeling needed — the Excel is used directly as ground truth.

```bash
./run_preprocess.sh
```

### 2. Fine-tune

See `finetune_qwen_vl_qa/FINETUNING_GUIDE.md` for the full training workflow.

### 3. Run with adapter

```bash
./start.sh --lora-adapter finetune_qwen_vl_qa/out/final_adapter
```

---

## Repository structure

```
ARCH-DOCUMENT-APP/
├── app/                              ← extraction app server + browser UI
│   ├── server.py
│   ├── excel_utils.py
│   └── static/
├── manual_redaction/                 ← manual redaction tool
│   ├── app.py
│   ├── static/
│   └── MANUAL_REDACTION_GUIDE.md
├── finetune_qwen_vl_qa/              ← QA-pair fine-tuning pipeline
│   ├── scripts/
│   │   ├── build_dataset.py          ← dataset builder
│   │   ├── qa_utils.py
│   │   └── serve_vlm_qa.py           ← VLM inference server
│   └── FINETUNING_GUIDE.md
├── finetune_qwen_vl_pytorch/         ← page-level pipeline (older approach)
├── start.sh                          ← start extraction tool
├── run_manual_redaction.sh           ← start redaction tool
├── run_preprocess.sh                 ← build fine-tuning dataset
├── PIPELINE_GUIDE.md                 ← page-level pipeline guide
├── QA_PIPELINE_GUIDE.md              ← QA-pair pipeline guide
└── README.md
```

---

## Prerequisites

- macOS with Apple Silicon — tested on M3 Ultra 96 GB
- Python 3.11

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv finetune_qwen_vl_pytorch/.venv311
source finetune_qwen_vl_pytorch/.venv311/bin/activate
pip install -r finetune_qwen_vl_pytorch/requirements.txt
pip install -U "git+https://github.com/huggingface/transformers"
pip install openpyxl
```

PDFs go in `Project Files/` — they are not tracked in git.
