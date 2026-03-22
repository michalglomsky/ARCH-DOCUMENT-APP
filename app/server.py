from __future__ import annotations

"""
App server for ARCH Document Extractor — port 8000.

Serves the browser UI and proxies requests to the VLM server (port 8081).

Start:
    source finetune_qwen_vl_pytorch/.venv311/bin/activate
    pip install fastapi uvicorn httpx openpyxl
    cd arch-document-app
    python app/server.py

Then open: http://localhost:8000
"""

import json
import sys
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))
from excel_utils import compare_result, parse_labels, save_result

BACKUP_DIR = Path("/Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup")
VLM_URL    = "http://127.0.0.1:8081"

app = FastAPI(title="ARCH Document Extractor")

# ---------------------------------------------------------------------------
# Serve static files (UI)
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/")
def root():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


# ---------------------------------------------------------------------------
# Document listing
# ---------------------------------------------------------------------------

@app.get("/api/documents")
def list_documents():
    """List all PDFs in the backup directory."""
    if not BACKUP_DIR.exists():
        return JSONResponse({"error": f"Backup directory not found: {BACKUP_DIR}"}, status_code=404)
    pdfs = sorted(BACKUP_DIR.glob("*.pdf"))
    return [{"name": p.name, "stem": p.stem} for p in pdfs]


# ---------------------------------------------------------------------------
# Serve PDF for the browser viewer
# ---------------------------------------------------------------------------

@app.get("/api/pdf/{filename}")
def serve_pdf(filename: str):
    """Serve a PDF file for PDF.js."""
    if not filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")
    path = BACKUP_DIR / filename
    if not path.is_file():
        raise HTTPException(404, f"PDF not found: {filename}")
    return FileResponse(str(path), media_type="application/pdf")


# ---------------------------------------------------------------------------
# Excel labels
# ---------------------------------------------------------------------------

_labels_cache: dict | None = None


def _get_labels() -> dict:
    global _labels_cache
    if _labels_cache is None:
        _labels_cache = parse_labels()
    return _labels_cache


@app.get("/api/labels/{nr}")
def get_label(nr: str):
    """Return the Excel ground-truth record for a given permit number."""
    labels = _get_labels()
    record = labels.get(nr)
    if not record:
        raise HTTPException(404, f"No Excel label found for nr_wniosku={nr}")
    return record


@app.get("/api/labels")
def list_labeled_nrs():
    """Return all permit numbers that have Excel labels."""
    return list(_get_labels().keys())


# ---------------------------------------------------------------------------
# VLM proxy — health
# ---------------------------------------------------------------------------

@app.get("/api/vlm/health")
async def vlm_health():
    """Check if the VLM server is running."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{VLM_URL}/health")
            return r.json()
    except Exception as e:
        return {"status": "offline", "error": str(e)}


# ---------------------------------------------------------------------------
# VLM proxy — extract
# ---------------------------------------------------------------------------

@app.post("/api/extract")
async def extract(body: dict):
    """
    Extract structured data from a PDF via the VLM server.
    Body: {"pdf_name": "wz_808.pdf", "max_pages": 6}
    """
    pdf_name = body.get("pdf_name", "")
    pdf_path = BACKUP_DIR / pdf_name
    if not pdf_path.is_file():
        raise HTTPException(404, f"PDF not found: {pdf_name}")

    payload = {"pdf_path": str(pdf_path), "max_pages": body.get("max_pages", 6)}
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{VLM_URL}/extract", json=payload)
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(503, "VLM server is not running. Start it with serve_vlm_qa.py --port 8081")


# ---------------------------------------------------------------------------
# VLM proxy — chat / free-text query
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(body: dict):
    """
    Send a free-text question about the current document to the VLM.
    Body: {"pdf_name": "wz_808.pdf", "message": "What is the roof type?", "max_pages": 6}
    Returns: {"response": "..."}
    """
    pdf_name = body.get("pdf_name", "")
    message  = body.get("message", "").strip()
    if not message:
        raise HTTPException(400, "message is required")

    pdf_path = BACKUP_DIR / pdf_name
    if not pdf_path.is_file():
        raise HTTPException(404, f"PDF not found: {pdf_name}")

    payload = {
        "pdf_path":  str(pdf_path),
        "message":   message,
        "max_pages": body.get("max_pages", 6),
    }
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{VLM_URL}/query", json=payload)
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(503, "VLM server is not running.")


# ---------------------------------------------------------------------------
# Compare VLM output with Excel ground truth
# ---------------------------------------------------------------------------

@app.post("/api/compare")
def compare(body: dict):
    """
    Compare a VLM prediction against the Excel label.
    Body: {"nr_wniosku": "808", "prediction": {...}}
    Returns field-by-field match/mismatch with overall accuracy.
    """
    nr   = str(body.get("nr_wniosku", ""))
    pred = body.get("prediction", {})
    labels = _get_labels()
    gold = labels.get(nr)
    if not gold:
        raise HTTPException(404, f"No Excel label for nr_wniosku={nr}")
    return compare_result(pred, gold)


# ---------------------------------------------------------------------------
# Save result to Excel
# ---------------------------------------------------------------------------

@app.post("/api/save")
def save(body: dict):
    """
    Append a VLM extraction result to extracted_results.xlsx.
    Body: {"prediction": {...}}
    """
    pred = body.get("prediction", {})
    if not pred:
        raise HTTPException(400, "prediction is required")
    out_path = save_result(pred)
    return {"status": "saved", "path": out_path}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    print(f"Starting ARCH Document Extractor at http://localhost:{args.port}")
    print(f"VLM server expected at: {VLM_URL}")
    print(f"PDF source: {BACKUP_DIR}")
    uvicorn.run("server:app", host="127.0.0.1", port=args.port, reload=args.reload,
                app_dir=str(Path(__file__).parent))
