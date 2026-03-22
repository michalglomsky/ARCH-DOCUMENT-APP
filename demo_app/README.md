## Document Extraction Demo (API vs Local VLM)

This demo app processes a folder of PDFs and extracts a fixed schema into a table (CSV/XLSX).
It is designed so you can **switch providers**:

- **Nanonets (API)**: run extraction using Nanonets workflows / extraction endpoint(s)
- **Local VLM**: call a local HTTP endpoint (e.g. `llama.cpp` server running Qwen2.5‑VL)

It can optionally run the local `redaction_tool/` first to remove private data.

### Setup

```bash
cd demo_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy config:

```bash
cp config.example.yaml config.yaml
```

### Run (CLI)

```bash
source demo_app/.venv/bin/activate
python -m extractor_demo.cli \
  --input-dir "Project Files" \
  --output-file "out.xlsx" \
  --config "demo_app/config.yaml"
```

### Output

- `out.xlsx` (or `.csv`) with one row per PDF.
- A `needs_review` column for low-confidence / unreadable fields.

