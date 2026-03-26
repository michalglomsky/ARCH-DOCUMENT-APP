from __future__ import annotations

"""
Manual Redaction Tool — FastAPI server (port 8083)

Draw redaction boxes on PDF pages via a browser UI.
Zones are stored as normalized coordinates (0-1) so they
transpose correctly across documents with different page sizes.

API:
  GET  /                   → UI
  POST /render             → {path} → rendered page images
  POST /list_pdfs          → {dir_path} → sorted list of PDF filenames
  POST /redact_save        → {path, out_path, zones} → apply & save
  POST /batch_redact       → {dir_path, out_dir, docs} → batch apply & save
"""

import base64
import sys
from pathlib import Path
from typing import Any

import fitz
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_ocr_reader = None  # lazy-loaded EasyOCR reader

_static = Path(__file__).parent / "static"

app = FastAPI(title="Manual Redaction Tool")
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RenderRequest(BaseModel):
    path: str
    dpi: int = 150


class ListPdfsRequest(BaseModel):
    dir_path: str


class BrowseRequest(BaseModel):
    path: str = ""


class Zone(BaseModel):
    x: float   # normalised 0-1 (left edge)
    y: float   # normalised 0-1 (top edge)
    w: float   # normalised width
    h: float   # normalised height


class RedactSaveRequest(BaseModel):
    path: str
    out_path: str
    zones: dict[str, list[Zone]]   # keys = page index as string
    pages_to_remove: list[int] = []


class BatchRedactRequest(BaseModel):
    dir_path: str
    out_dir: str
    docs: dict[str, dict[str, list[Zone]]]   # {filename: {page_idx: [zones]}}
    pages_to_remove: dict[str, list[int]] = {}   # {filename: [page_indices]}


class AnchorRegion(BaseModel):
    x: float   # normalised 0-1
    y: float
    w: float
    h: float


class PageAnchor(BaseModel):
    x: float   # normalised region to search within
    y: float
    w: float
    h: float
    text: str  # anchor text(s), comma-separated


class AlignRequest(BaseModel):
    template_doc_path: str
    template_zones: dict[str, list[Zone]]   # page_idx_str → zones
    target_doc_path: str
    page_anchors: dict[str, PageAnchor]     # page_idx_str → anchor (per-page)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_pdf(pdf_path: Path, dpi: int) -> list[dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpeg")
        b64 = base64.b64encode(img_bytes).decode()
        pages.append({
            "index": i,
            "width_pt": page.rect.width,
            "height_pt": page.rect.height,
            "image_b64": b64,
        })
    doc.close()
    return pages


def _apply_zones(
    pdf_path: Path,
    out_path: Path,
    zones: dict[str, list[Zone]],
    pages_to_remove: list[int] | None = None,
) -> int:
    """Apply normalised zones to PDF and save to out_path. Returns total redactions applied."""
    remove_set = set(pages_to_remove or [])
    doc = fitz.open(str(pdf_path))
    total = 0
    for page_idx_str, zone_list in zones.items():
        if not zone_list:
            continue
        try:
            page_idx = int(page_idx_str)
        except ValueError:
            continue
        if page_idx >= doc.page_count or page_idx in remove_set:
            continue
        page = doc.load_page(page_idx)
        w_pt = page.rect.width
        h_pt = page.rect.height
        for z in zone_list:
            x0 = z.x * w_pt
            y0 = z.y * h_pt
            x1 = (z.x + z.w) * w_pt
            y1 = (z.y + z.h) * h_pt
            page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(0, 0, 0))
            total += 1
        # images=0: draw black fill but don't remove underlying image data.
        # PDF_REDACT_IMAGE_REMOVE can corrupt image-based PDFs by removing
        # more image content than expected.
        page.apply_redactions(images=0)
    # Delete pages in reverse order to preserve indices
    for i in sorted(remove_set, reverse=True):
        if 0 <= i < doc.page_count:
            doc.delete_page(i)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    return total


def _find_anchor_text(
    pdf_path: Path, page_idx: int, anchor_texts: list[str],
    region: AnchorRegion | None = None,
) -> dict | None:
    """PyMuPDF built-in text search, optionally clipped to a normalised region."""
    doc = fitz.open(str(pdf_path))
    if page_idx >= doc.page_count:
        doc.close()
        return None
    page = doc.load_page(page_idx)
    w, h = page.rect.width, page.rect.height
    clip = None
    if region:
        clip = fitz.Rect(region.x * w, region.y * h,
                         (region.x + region.w) * w, (region.y + region.h) * h)
    for text in anchor_texts:
        for variant in (text, text.lower(), text.upper(), text.title()):
            rects = page.search_for(variant, clip=clip) if clip else page.search_for(variant)
            if rects:
                r = rects[0]
                doc.close()
                return {"x": r.x0 / w, "y": r.y0 / h, "found_text": variant}
    doc.close()
    return None


def _find_anchor_ocr(
    pdf_path: Path, page_idx: int, anchor_texts: list[str],
    region: AnchorRegion | None = None, dpi: int = 150,
) -> dict | None:
    """EasyOCR fallback — crops to region when provided (much faster)."""
    global _ocr_reader
    try:
        import numpy as np
        import easyocr
        if _ocr_reader is None:
            _ocr_reader = easyocr.Reader(["pl", "en"], gpu=True)
        doc = fitz.open(str(pdf_path))
        if page_idx >= doc.page_count:
            doc.close()
            return None
        page = doc.load_page(page_idx)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        doc.close()
        iw, ih = pix.width, pix.height
        # Crop to region if provided — limits OCR to a small area
        ox, oy = 0, 0
        if region:
            ox = max(0, int(region.x * iw))
            oy = max(0, int(region.y * ih))
            ex = min(iw, int((region.x + region.w) * iw))
            ey = min(ih, int((region.y + region.h) * ih))
            img = img[oy:ey, ox:ex]
        results = _ocr_reader.readtext(img)
        for anchor in anchor_texts:
            for bbox, text, conf in results:
                if anchor.lower() in text.lower() and conf > 0.3:
                    # bbox coords are relative to the crop — convert back to full page
                    x0 = (ox + bbox[0][0]) / iw
                    y0 = (oy + bbox[0][1]) / ih
                    return {"x": x0, "y": y0, "found_text": text.strip()}
        return None
    except Exception:
        return None


def _locate_anchor(
    pdf_path: Path, page_idx: int, anchor_texts: list[str],
    region: AnchorRegion | None = None,
) -> dict | None:
    """Try text search first, fall back to OCR."""
    result = _find_anchor_text(pdf_path, page_idx, anchor_texts, region)
    if result is None:
        result = _find_anchor_ocr(pdf_path, page_idx, anchor_texts, region)
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(str(_static / "index.html"))


@app.post("/browse")
def browse(req: BrowseRequest):
    """List directory contents for the file explorer."""
    import os
    p = Path(req.path).expanduser() if req.path else Path.home()
    if not p.exists():
        p = Path.home()
    if not p.is_dir():
        p = p.parent
    p = p.resolve()

    dirs, files = [], []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": str(entry)})
            elif entry.suffix.lower() == ".pdf":
                files.append({"name": entry.name, "path": str(entry)})
    except PermissionError:
        pass

    parent = str(p.parent) if p != p.parent else None
    return {"current": str(p), "parent": parent, "dirs": dirs, "files": files}


@app.post("/mkdir")
def mkdir(req: BrowseRequest):
    """Create a directory (and any missing parents)."""
    p = Path(req.path).expanduser().resolve()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    return {"path": str(p)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/render")
def render(req: RenderRequest):
    pdf = Path(req.path)
    if not pdf.is_file():
        raise HTTPException(404, f"File not found: {req.path}")
    pages = _render_pdf(pdf, req.dpi)
    return {"path": str(pdf), "name": pdf.name, "pages": pages}


@app.post("/list_pdfs")
def list_pdfs(req: ListPdfsRequest):
    d = Path(req.dir_path)
    if not d.is_dir():
        raise HTTPException(404, f"Directory not found: {req.dir_path}")
    pdfs = sorted(p.name for p in d.glob("*.pdf"))
    return {"dir_path": str(d), "files": pdfs}


@app.post("/redact_save")
def redact_save(req: RedactSaveRequest):
    pdf = Path(req.path)
    if not pdf.is_file():
        raise HTTPException(404, f"File not found: {req.path}")
    out = Path(req.out_path)
    n = _apply_zones(pdf, out, req.zones, pages_to_remove=req.pages_to_remove or None)
    return {"status": "ok", "out_path": str(out), "redactions_applied": n}


@app.post("/align_zones")
def align_zones(req: AlignRequest):
    """
    For each page that has an anchor, locate the anchor text in both the template
    and target documents and compute a per-page translation offset.
    Pages without an anchor are copied directly without shifting.
    """
    tmpl_path = Path(req.template_doc_path)
    tgt_path  = Path(req.target_doc_path)
    if not tmpl_path.is_file():
        raise HTTPException(404, f"Template doc not found: {req.template_doc_path}")
    if not tgt_path.is_file():
        raise HTTPException(404, f"Target doc not found: {req.target_doc_path}")

    # Build plain-dict zones from the request (Zone objects → dicts)
    raw_zones: dict[str, list[dict]] = {
        k: [z.model_dump() for z in zl]
        for k, zl in req.template_zones.items()
    }

    if not req.page_anchors:
        return {"aligned": False, "aligned_pages": 0, "fallback_pages": 0,
                "zones": raw_zones, "detail": "No anchors set — zones copied without shift"}

    shifted: dict[str, list[dict]] = {}
    aligned_pages = 0
    fallback_pages = 0

    for page_str, zone_list in raw_zones.items():
        anchor = req.page_anchors.get(page_str)
        if anchor:
            page_idx = int(page_str)
            anchor_texts = [t.strip() for t in anchor.text.split(",") if t.strip()]
            region = AnchorRegion(x=anchor.x, y=anchor.y, w=anchor.w, h=anchor.h)
            tmpl_pt = _locate_anchor(tmpl_path, page_idx, anchor_texts, region)
            tgt_pt  = _locate_anchor(tgt_path,  page_idx, anchor_texts, region)
            if tmpl_pt and tgt_pt:
                dx = tgt_pt["x"] - tmpl_pt["x"]
                dy = tgt_pt["y"] - tmpl_pt["y"]
                shifted[page_str] = [
                    {
                        "x": max(0.0, min(1.0 - z["w"], z["x"] + dx)),
                        "y": max(0.0, min(1.0 - z["h"], z["y"] + dy)),
                        "w": z["w"],
                        "h": z["h"],
                    }
                    for z in zone_list
                ]
                aligned_pages += 1
            else:
                # anchor not found in one of the docs — copy without shift
                shifted[page_str] = zone_list
                fallback_pages += 1
        else:
            # no anchor for this page — copy directly
            shifted[page_str] = zone_list

    return {
        "aligned": aligned_pages > 0,
        "aligned_pages": aligned_pages,
        "fallback_pages": fallback_pages,
        "zones": shifted,
    }


@app.post("/batch_redact")
def batch_redact(req: BatchRedactRequest):
    src_dir = Path(req.dir_path)
    out_dir = Path(req.out_dir)
    if not src_dir.is_dir():
        raise HTTPException(404, f"Directory not found: {req.dir_path}")

    results = []
    for filename, zones in req.docs.items():
        src = src_dir / filename
        out = out_dir / filename
        if not src.is_file():
            results.append({"file": filename, "status": "error", "detail": "not found"})
            continue
        try:
            remove = req.pages_to_remove.get(filename) or None
            n = _apply_zones(src, out, zones, pages_to_remove=remove)
            results.append({"file": filename, "status": "ok", "redactions_applied": n})
        except Exception as exc:
            results.append({"file": filename, "status": "error", "detail": str(exc)})

    ok = sum(1 for r in results if r["status"] == "ok")
    return {"processed": len(results), "ok": ok, "results": results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8083)
    args = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
