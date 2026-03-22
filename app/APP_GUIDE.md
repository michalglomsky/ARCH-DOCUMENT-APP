# App Layer Guide — ARCH Document Extractor

Browser-based UI for running VLM extraction, comparing against Excel ground truth, and chatting with the model about a document.

---

## Architecture

```
Browser  ──▶  app/server.py (port 8000)  ──▶  serve_vlm_qa.py (port 8081)
               ├── serves static UI                 └── Qwen2.5-VL on MPS
               ├── proxies /api/extract → /extract
               ├── proxies /api/chat    → /query
               └── reads Excel directly (excel_utils.py)
```

The app server never loads the VLM model — it proxies all inference to the separate VLM server on port 8081.  This means you can restart the UI server without reloading the 14 GB model.

---

## File layout

```
app/
├── server.py          ← FastAPI app server (port 8000)
├── excel_utils.py     ← parse/compare/save Excel helpers
├── requirements.txt
├── APP_GUIDE.md       ← this file
└── static/
    ├── index.html     ← single-page UI
    ├── app.js         ← PDF.js viewer + API calls
    └── style.css      ← dark theme layout
```

---

## Quick start

### 1. Start the VLM server (port 8081)

```bash
source finetune_qwen_vl_pytorch/.venv311/bin/activate

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

Wait for: `Model ready on mps.`

### 2. Start the app server (port 8000)

In a **separate terminal** (same venv, or a fresh one):

```bash
source finetune_qwen_vl_pytorch/.venv311/bin/activate
pip install fastapi uvicorn httpx openpyxl   # one-time

cd arch-document-app
python app/server.py
```

### 3. Open the browser

```
http://localhost:8000
```

---

## UI features

### Document selector (top bar)
Dropdown lists all PDFs from the backup directory.  Select one to load it in the PDF viewer.

### PDF viewer (left pane)
PDF.js renders the document page by page.  Use ◀ ▶ arrows to navigate.  The viewer is read-only — pages are served directly from the backup folder.

### Max pages (top bar)
Controls how many pages are sent to the VLM for extraction and chat.  Default 6.  Reduce to 4 for faster inference; increase to 10 for longer documents.

### VLM status indicator (top right)
Shows whether the VLM server on port 8081 is reachable.  Auto-refreshes every 30 s.  Displays device (mps/cpu) and whether a LoRA adapter is loaded.

---

## Tabs (right pane)

### Extraction tab
| Button | Action |
|--------|--------|
| **⚡ Extract** | Sends PDF to VLM, renders structured result |
| **⚖ Compare** | Runs field-by-field diff against Excel ground truth; switches to Compare tab |
| **💾 Save** | Appends prediction to `app/extracted_results.xlsx` |
| **🔄 Re-extract** | Runs extraction again (useful after changing Max pages) |

The extraction result shows:
- Flat fields (nr_wniosku, adres, …)
- Per-building section (budynki list)
- Media list
- `⚠ needs_review = true` warning if the model flagged uncertainty

### Compare tab
Shows each field with **PRED** (VLM output) and **GOLD** (Excel label) side by side.
- Green left border = match
- Red left border = mismatch
- Accuracy bar at the top shows overall percentage

Requires the permit number (`nr_wniosku`) from the extraction to exist in `1-2026-DANE.xlsx`.

### Chat tab
Send any free-text question about the current document to the VLM.  All pages (up to Max pages) are sent as context.  Useful for:
- "What is the investment name?"
- "How many floors does building 2 have?"
- "Is there any basement?"
- "Summarise the media connections."

Press **Enter** to send (Shift+Enter for newline).

---

## API reference (server.py)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves index.html |
| GET | `/api/documents` | Lists all PDFs in backup directory |
| GET | `/api/pdf/{filename}` | Serves PDF file for PDF.js |
| GET | `/api/labels/{nr}` | Returns Excel ground-truth record for permit number |
| GET | `/api/labels` | Returns list of all permit numbers with Excel labels |
| GET | `/api/vlm/health` | Checks if VLM server is running |
| POST | `/api/extract` | `{pdf_name, max_pages}` → VLM extraction result |
| POST | `/api/chat` | `{pdf_name, message, max_pages}` → `{response}` |
| POST | `/api/compare` | `{nr_wniosku, prediction}` → field-by-field comparison |
| POST | `/api/save` | `{prediction}` → appends to extracted_results.xlsx |

---

## Excel output

`app/extracted_results.xlsx` is created on the first Save.  Each save appends rows for the current prediction.  The file uses the same column layout as `1-2026-DANE.xlsx` and `EXTRACTED-DATA-TEMPLATE.xlsx`.

---

## Troubleshooting

### "VLM: ✗ offline"
The VLM server on port 8081 is not running.  Start it with `serve_vlm_qa.py`.

### "No Excel label found for nr_wniosku=XXX"
The permit number extracted by the VLM does not match any row in `1-2026-DANE.xlsx`.  Check that the extraction is correct, or use a different document.

### PDF not loading
The PDF must exist in `/Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup/`.  Check `BACKUP_DIR` in `app/server.py` if you've moved the files.

### Extract returns `_parse_error`
The VLM output could not be parsed as JSON.  Usually happens in zero-shot mode on unusual forms.  Fine-tuning with examples of that form type will fix this.

### Port 8000 already in use
Change the port: `python app/server.py --port 8001`
