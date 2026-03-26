# ARCH-DOCUMENT-APP

Automated extraction of structured data from Polish construction permit application forms (wnioski o warunki zabudowy / pozwolenie na budowę) using a locally-run Vision Language Model (Qwen2.5-VL-7B) fine-tuned with LoRA.

Runs on **Apple Silicon** (MPS) or **Windows/Linux with an NVIDIA GPU** (CUDA).

---

## Tools

### Extraction tool — `./start.sh` (macOS) or Docker

Browser UI that sends scanned PDFs to a local VLM and extracts structured data into an Excel spreadsheet.

- Extracts: permit number, fill method, investment name, address, plot description, built-up area, facade width, floor counts, building height, roof geometry, utilities
- Output maps directly to the master Excel spreadsheet

### Manual redaction tool — `./run_manual_redaction.sh` (macOS) or Docker

Browser UI for drawing redaction boxes over PDF pages and saving redacted copies.

- Renders PDF pages in the browser; click and drag to draw redaction zones
- Supports batch redaction across a whole directory of PDFs
- Redacted files are saved with a `.redacted.pdf` suffix

See `manual_redaction/MANUAL_REDACTION_GUIDE.md` for full usage.

---

## Running with Docker (Windows / Linux)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend (Windows)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support
- NVIDIA GPU with at least 16 GB VRAM recommended (the model is ~15 GB)

### 1. Set your PDF folder path

Create a `.env` file in the project root:

```env
PDF_DIR=C:\Users\yourname\Documents\arch-pdfs
```

On Linux:
```env
PDF_DIR=/home/yourname/arch-pdfs
```

The folder should contain your `.pdf` files and the `1-2026-DANE.xlsx` labels spreadsheet.

### 2. Build and start

```bash
docker compose up --build
```

The first run downloads the Qwen2.5-VL-7B model (~15 GB) into a named Docker volume — this only happens once.

### 3. Open in browser

| Tool | URL |
|------|-----|
| Extraction UI | http://localhost:8000 |
| Manual redaction | http://localhost:8083 |

### Running individual tools

Start only the redaction tool (no GPU needed):

```bash
docker compose up --build redaction
```

Start only the extraction stack:

```bash
docker compose up --build vlm-server app-server
```

### Using a fine-tuned LoRA adapter

Pass the adapter path as a command override:

```bash
docker compose run vlm-server python /app/scripts/serve_vlm_qa.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --lora-adapter /data/out/final_adapter \
    --port 8081
```

---

## Running on macOS (Apple Silicon)

### Prerequisites

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

### Start extraction tool

```bash
./start.sh                              # zero-shot model
./start.sh --lora-adapter path/to/out   # with LoRA fine-tuned adapter
./start.sh --port-app 8001              # custom port
./start.sh --max-pages 4                # limit pages per inference
```

Open `http://localhost:8000` in your browser.

### Start redaction tool

```bash
./run_manual_redaction.sh
```

Open `http://localhost:8083` in your browser.

---

## Fine-tuning pipeline

The QA-pair pipeline fine-tunes the model on your own labeled documents.

### 1. Build the dataset

```bash
./run_preprocess.sh
```

Renders PDF pages to images and builds `train.jsonl` / `val.jsonl` / `test.jsonl` from `1-2026-DANE.xlsx`. No manual labeling needed.

### 2. Fine-tune

See `finetune_qwen_vl_qa/FINETUNING_GUIDE.md` for the full training workflow.

### 3. Run with adapter

```bash
# macOS
./start.sh --lora-adapter finetune_qwen_vl_qa/out/final_adapter

# Docker
docker compose run vlm-server python /app/scripts/serve_vlm_qa.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --lora-adapter /data/out/final_adapter \
    --port 8081
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
│   │   ├── build_dataset.py
│   │   ├── qa_utils.py
│   │   └── serve_vlm_qa.py           ← VLM inference server
│   └── FINETUNING_GUIDE.md
├── finetune_qwen_vl_pytorch/         ← page-level pipeline (older approach)
├── requirements/
│   ├── vlm.txt                       ← VLM server dependencies
│   ├── app.txt                       ← app server dependencies
│   └── redaction.txt                 ← redaction tool dependencies
├── Dockerfile.vlm                    ← VLM server image (CUDA)
├── Dockerfile.app                    ← app server image
├── Dockerfile.redaction              ← redaction tool image
├── docker-compose.yml                ← orchestrates all three services
├── start.sh                          ← macOS: start extraction tool
├── run_manual_redaction.sh           ← macOS: start redaction tool
├── run_preprocess.sh                 ← build fine-tuning dataset
├── PIPELINE_GUIDE.md
├── QA_PIPELINE_GUIDE.md
└── README.md
```

PDFs go in your local folder — they are never committed to git.
