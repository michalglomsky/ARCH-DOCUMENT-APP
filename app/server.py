from __future__ import annotations

"""
App server for ARCH Document Extractor — port 8000.

Serves the browser UI and proxies requests to the VLM server (port 8081).

Start:
    cd arch-document-app
    ./start.sh
    # or manually:
    PDF_DIR=/path/to/pdfs python app/server.py

Then open: http://localhost:8000

Config can also be changed live via POST /api/config.
"""

import json
import os
import sys
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))
from excel_utils import compare_result, parse_labels, save_result, get_excel_schema

# ---------------------------------------------------------------------------
# Runtime config — overridable via env vars or POST /api/config
# ---------------------------------------------------------------------------

_DEFAULT_PDF_DIR   = "/Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup"
_DEFAULT_LABELS_XL = "/Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup/1-2026-DANE.xlsx"

_config: dict = {
    "pdf_dir":    os.environ.get("PDF_DIR",    _DEFAULT_PDF_DIR),
    "labels_xlsx": os.environ.get("LABELS_XLSX", _DEFAULT_LABELS_XL),
    "vlm_url":    os.environ.get("VLM_URL",    "http://127.0.0.1:8081"),
}

# Cache invalidated whenever labels_xlsx changes
_labels_cache: dict | None = None


def _pdf_dir()    -> Path: return Path(_config["pdf_dir"])
def _vlm_url()    -> str:  return _config["vlm_url"]
def _labels_xlsx()-> Path: return Path(_config["labels_xlsx"])


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ARCH Document Extractor")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/")
def root():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    """Return current server config."""
    return {**_config, "pdf_dir_exists": _pdf_dir().exists()}


@app.post("/api/config")
def set_config(body: dict):
    """
    Update runtime config.
    Body: {"pdf_dir": "/new/path", "labels_xlsx": "/new/labels.xlsx", "vlm_url": "http://..."}
    Only provided keys are updated.
    """
    global _labels_cache
    changed = []
    for key in ("pdf_dir", "labels_xlsx", "vlm_url"):
        if key in body:
            _config[key] = body[key]
            changed.append(key)
    if "labels_xlsx" in changed or "pdf_dir" in changed:
        _labels_cache = None  # force reload
    return {"updated": changed, **_config}


# ---------------------------------------------------------------------------
# Filesystem browser (for directory picker)
# ---------------------------------------------------------------------------

@app.get("/api/browse")
def browse(path: str = "/"):
    """
    List subdirectories and PDF count at the given path.
    Used by the directory picker in the UI.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, f"Directory not found: {path}")
    try:
        entries = list(p.iterdir())
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {path}")

    dirs = sorted(
        [e for e in entries if e.is_dir() and not e.name.startswith(".")],
        key=lambda x: x.name.lower(),
    )
    pdf_count = sum(1 for e in entries if e.suffix.lower() == ".pdf")

    return {
        "path":      str(p),
        "parent":    str(p.parent) if p.parent != p else None,
        "dirs":      [{"name": d.name, "path": str(d)} for d in dirs],
        "pdf_count": pdf_count,
    }


# ---------------------------------------------------------------------------
# Document listing
# ---------------------------------------------------------------------------

@app.get("/api/documents")
def list_documents():
    """List all PDFs in the configured pdf_dir."""
    d = _pdf_dir()
    if not d.exists():
        return JSONResponse({"error": f"PDF directory not found: {d}"}, status_code=404)
    pdfs = sorted(d.glob("*.pdf"))
    return [{"name": p.name, "stem": p.stem} for p in pdfs]


# ---------------------------------------------------------------------------
# Serve PDF for the browser viewer
# ---------------------------------------------------------------------------

@app.get("/api/pdf/{filename}")
def serve_pdf(filename: str):
    """Serve a PDF file for PDF.js."""
    if not filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")
    path = _pdf_dir() / filename
    if not path.is_file():
        raise HTTPException(404, f"PDF not found: {filename}")
    return FileResponse(str(path), media_type="application/pdf")


# ---------------------------------------------------------------------------
# Excel labels
# ---------------------------------------------------------------------------

def _get_labels() -> dict:
    global _labels_cache
    if _labels_cache is None:
        _labels_cache = parse_labels(_labels_xlsx())
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
# Schema — fields defined in an Excel template
# ---------------------------------------------------------------------------

@app.get("/api/schema")
def get_schema():
    """
    Return the extraction field schema derived from the labels Excel or
    EXTRACTED-DATA-TEMPLATE.xlsx.  Used by the UI to render the grid columns.
    """
    return get_excel_schema()


# ---------------------------------------------------------------------------
# VLM proxy — health
# ---------------------------------------------------------------------------

@app.get("/api/vlm/health")
async def vlm_health():
    """Check if the VLM server is running."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{_vlm_url()}/health")
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
    pdf_path = _pdf_dir() / pdf_name
    if not pdf_path.is_file():
        raise HTTPException(404, f"PDF not found: {pdf_name}")

    payload = {"pdf_path": str(pdf_path), "max_pages": body.get("max_pages", 6)}
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{_vlm_url()}/extract", json=payload)
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

    pdf_path = _pdf_dir() / pdf_name
    if not pdf_path.is_file():
        raise HTTPException(404, f"PDF not found: {pdf_name}")

    payload = {
        "pdf_path":  str(pdf_path),
        "message":   message,
        "max_pages": body.get("max_pages", 6),
    }
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{_vlm_url()}/query", json=payload)
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
    ap.add_argument("--port",   type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    print(f"Starting ARCH Document Extractor at http://localhost:{args.port}")
    print(f"VLM server expected at: {_vlm_url()}")
    print(f"PDF source: {_pdf_dir()}")
    uvicorn.run("server:app", host="127.0.0.1", port=args.port, reload=args.reload,
                app_dir=str(Path(__file__).parent))
