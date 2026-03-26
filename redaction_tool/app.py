from __future__ import annotations

"""
FastAPI server for the PDF Redaction Tool.

Usage:
    pip install -r requirements.txt
    python redaction_tool/app.py --port 8082

API:
    GET  /           → serve UI
    GET  /health     → {"status": "ok", ...}
    POST /preview    → upload PDF, get annotated page images + stats (dry-run)
    POST /redact     → upload PDF, download redacted PDF
"""

import argparse
import base64
import sys
import tempfile
from pathlib import Path

import cv2
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))
from redaction_tool.redactor import _load_config, preview_pdf, redact_pdf

try:
    import easyocr
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("easyocr not installed. Run: pip install -r requirements.txt") from exc


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_reader = None
_cfg = None
_config_path: Path = Path(__file__).parent / "config.yaml"


def _get_reader():
    global _reader
    if _reader is None:
        raise HTTPException(status_code=503, detail="OCR reader not initialised yet.")
    return _reader


def _get_cfg():
    global _cfg
    if _cfg is None:
        raise HTTPException(status_code=503, detail="Config not loaded yet.")
    return _cfg


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PDF Redaction Tool")

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.on_event("startup")
def startup() -> None:
    global _reader, _cfg
    if not _config_path.exists():
        raise RuntimeError(f"Config not found: {_config_path}")
    _cfg = _load_config(_config_path)
    print(f"Loading EasyOCR ({_cfg.ocr_langs}) on CPU …")
    _reader = easyocr.Reader(_cfg.ocr_langs, gpu=True)
    print("Ready.")


@app.get("/")
def index():
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/health")
def health():
    cfg = _get_cfg()
    return {
        "status": "ok",
        "config": str(_config_path),
        "ocr_langs": cfg.ocr_langs,
        "render_dpi": cfg.render_dpi,
        "page_types": [pt.name for pt in cfg.page_types],
    }


@app.post("/preview")
async def preview(file: UploadFile = File(...), show_ocr_boxes: bool = False):
    """
    Dry-run: render each page with redaction zones highlighted in red.
    Returns JSON with per-page annotated images (base64 JPEG) and stats.
    Does NOT modify the uploaded PDF.
    """
    cfg = _get_cfg()
    reader = _get_reader()

    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        pages_raw = preview_pdf(tmp_path, cfg, reader, show_ocr_boxes=show_ocr_boxes)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    pages_out = []
    for p in pages_raw:
        img: "cv2.typing.MatLike" = p["annotated_bgr"]
        # Scale down for transfer (max 1000px on longest side)
        h, w = img.shape[:2]
        scale = min(1.0, 1000.0 / max(h, w, 1))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(bytes(buf)).decode()
        pages_out.append({
            "page_index": p["page_index"],
            "page_type": p["page_type"],
            "redaction_count": p["redaction_count"],
            "matches": p.get("matches", []),
            "image_b64": b64,
        })

    return {"filename": file.filename, "pages": pages_out}


@app.post("/redact")
async def redact(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pages_to_remove: str = Form(""),
):
    """
    Apply redactions and return the redacted PDF as a download.
    pages_to_remove: comma-separated 0-based page indices to delete from the output.
    The original uploaded file is never stored permanently.
    """
    cfg = _get_cfg()
    reader = _get_reader()

    remove_indices: list[int] = []
    for part in pages_to_remove.split(","):
        part = part.strip()
        if part.isdigit():
            remove_indices.append(int(part))

    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
        tmp_in.write(data)
        tmp_in_path = Path(tmp_in.name)

    out_stem = Path(file.filename or "document").stem
    tmp_out_path = tmp_in_path.parent / f"{out_stem}.redacted.pdf"

    try:
        redact_pdf(tmp_in_path, tmp_out_path, cfg, reader, pages_to_remove=remove_indices or None)
    except Exception as exc:
        tmp_in_path.unlink(missing_ok=True)
        tmp_out_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_in_path.unlink(missing_ok=True)

    # Clean up the output file after the response is sent
    background_tasks.add_task(lambda: tmp_out_path.unlink(missing_ok=True))

    return FileResponse(
        path=str(tmp_out_path),
        media_type="application/pdf",
        filename=f"{out_stem}.redacted.pdf",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PDF Redaction Tool server")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--config", default=str(_config_path),
                    help="Path to config.yaml (default: redaction_tool/config.yaml)")
    args = ap.parse_args()
    _config_path = Path(args.config).expanduser().resolve()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
