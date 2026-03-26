from __future__ import annotations

"""
FastAPI inference server for the QA-pair VLM.

Accepts a PDF path (or list of pre-rendered image paths), runs multi-page
inference, and returns structured JSON.  Also supports direct Excel export.

Usage:
    source finetune_qwen_vl_pytorch/.venv311/bin/activate

    # Zero-shot:
    python finetune_qwen_vl_qa/scripts/serve_vlm_qa.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --port 8081

    # Fine-tuned adapter:
    python finetune_qwen_vl_qa/scripts/serve_vlm_qa.py \
        --model        Qwen/Qwen2.5-VL-7B-Instruct \
        --lora-adapter finetune_qwen_vl_qa/out/lora_run1 \
        --port 8081

API endpoints:
    GET  /health
         → {"status": "ok", "device": "mps", "model": "...", "adapter": "..."}

    POST /extract
         Body: {"pdf_path": "/abs/path/to/file.pdf", "max_pages": 6}
         → structured JSON record (nr_wniosku, budynki, media, …)

    POST /query
         Body: {"pdf_path": "/abs/path/to/file.pdf", "message": "What is the roof type?", "max_pages": 6}
         → {"response": "..."}

    POST /extract_batch
         Body: {"pdf_dir": "/abs/path/to/folder", "max_pages": 6}
         → list of structured JSON records, one per PDF

    POST /extract_to_xlsx
         Body: {"pdf_dir": "/abs/path/to/folder", "output_xlsx": "/abs/path/out.xlsx"}
         → {"status": "ok", "n_processed": N, "output_xlsx": "..."}
"""

import argparse
import json
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from qa_utils import (
    build_qa_prompt,
    parse_json_response,
    preprocess_pages,
    render_pdf_pages,
    render_pdf_to_tmp,
    run_inference_multipage,
)


# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------

_model: Any = None
_processor: Any = None
_device: str = "cpu"
_model_name: str = ""
_adapter_path: str = ""
_prompt: str = build_qa_prompt()
_max_pixels: int = 501_760
_max_pages: int = 6
_preprocess: bool = False          # enable adaptive preprocessing
_preprocess_debug_dir: str = ""    # save processed pages here when non-empty
_lock = threading.Lock()


def load_model(model_name: str, lora_adapter: str, max_pixels: int) -> None:
    global _model, _processor, _device, _model_name, _adapter_path, _max_pixels
    _device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading {model_name} on {_device} …")
    _processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
    )
    if lora_adapter:
        print(f"Applying adapter: {lora_adapter}")
        _model = PeftModel.from_pretrained(_model, lora_adapter)
        _adapter_path = lora_adapter
    _model.to(_device)
    _model.eval()
    _model_name = model_name
    _max_pixels = max_pixels
    print(f"Model ready on {_device}.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="QA VLM Extraction Server")


class ExtractRequest(BaseModel):
    pdf_path: str
    max_pages: int = 6


class BatchRequest(BaseModel):
    pdf_dir: str
    max_pages: int = 6


class XlsxRequest(BaseModel):
    pdf_dir: str
    output_xlsx: str
    max_pages: int = 6


class QueryRequest(BaseModel):
    pdf_path: str
    message: str
    max_pages: int = 6


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": _device,
        "model": _model_name,
        "adapter": _adapter_path or "none (zero-shot)",
    }


def _run_pdf(pdf_path: Path, max_pages: int) -> dict:
    tmp_dir, page_paths = render_pdf_to_tmp(pdf_path, dpi=250, max_pages=max_pages)
    try:
        # Adaptive preprocessing (optional — enabled via --preprocess flag)
        if _preprocess:
            debug_dir = None
            if _preprocess_debug_dir:
                debug_dir = Path(_preprocess_debug_dir) / pdf_path.stem
            page_paths, pp_stats = preprocess_pages(page_paths, out_dir=debug_dir, enabled=True)
            if debug_dir and pp_stats:
                import json as _json
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / "preprocess_stats.json").write_text(
                    _json.dumps(pp_stats, indent=2, ensure_ascii=False)
                )
            if not page_paths:
                return {"needs_review": True, "_error": "all pages were blank after preprocessing"}

        with _lock:
            raw = run_inference_multipage(
                _model, _processor, page_paths, _prompt, _device, max_new_tokens=1024
            )
            if _device == "mps":
                torch.mps.empty_cache()
        result = parse_json_response(raw)
        result["_source_pdf"] = str(pdf_path)
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/extract")
def extract(req: ExtractRequest):
    pdf = Path(req.pdf_path)
    if not pdf.is_file():
        raise HTTPException(status_code=404, detail=f"PDF not found: {req.pdf_path}")
    return _run_pdf(pdf, req.max_pages)


@app.post("/query")
def query(req: QueryRequest):
    """Answer a free-text question about a document using all its pages as context."""
    pdf = Path(req.pdf_path)
    if not pdf.is_file():
        raise HTTPException(status_code=404, detail=f"PDF not found: {req.pdf_path}")
    tmp_dir, page_paths = render_pdf_to_tmp(pdf, dpi=250, max_pages=req.max_pages)
    try:
        with _lock:
            raw = run_inference_multipage(
                _model, _processor, page_paths, req.message, _device, max_new_tokens=512
            )
            if _device == "mps":
                torch.mps.empty_cache()
        return {"response": raw}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/extract_batch")
def extract_batch(req: BatchRequest):
    pdf_dir = Path(req.pdf_dir)
    if not pdf_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {req.pdf_dir}")
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    results = []
    for pdf in pdfs:
        try:
            results.append(_run_pdf(pdf, req.max_pages))
        except Exception as exc:
            results.append({"_source_pdf": str(pdf), "_error": str(exc)})
    return results


@app.post("/extract_to_xlsx")
def extract_to_xlsx(req: XlsxRequest):
    pdf_dir   = Path(req.pdf_dir)
    out_xlsx  = Path(req.output_xlsx)
    if not pdf_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {req.pdf_dir}")

    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    records = []
    for pdf in pdfs:
        try:
            records.append(_run_pdf(pdf, req.max_pages))
        except Exception as exc:
            records.append({"_source_pdf": str(pdf), "_error": str(exc)})

    _write_xlsx(records, out_xlsx)
    return {"status": "ok", "n_processed": len(records), "output_xlsx": str(out_xlsx)}


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------

def _write_xlsx(records: list[dict], output_path: Path) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Font
    except ImportError:
        raise RuntimeError("openpyxl not installed. Run: pip install openpyxl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Extracted Data"

    headers = [
        "Nr wniosku", "Sposób wypełnienia", "Flaga 7.9", "Nazwa inwestycji",
        "Adres", "Teren inwestycji", "Pow. zabudowy (całość)",
        "Szerokość elewacji", "Suma pow. nadziemnych", "Suma pow. podziemnych",
        "Wys. górnej krawędzi", "Wysokość zabudowy",
        "Ilość kond. nadziemnych", "Ilość kond. podziemnych",
        "Geometria dachu", "Media",
    ]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for pred in records:
        if "_error" in pred:
            ws.append([pred.get("_source_pdf", ""), pred["_error"]])
            continue

        nr      = pred.get("nr_wniosku", "")
        sposob  = pred.get("sposob_wypelnienia", "")
        flaga   = pred.get("flaga_7_9", "")
        nazwa   = pred.get("nazwa_inwestycji", "")
        adres   = pred.get("adres", "")
        teren   = pred.get("teren_inwestycji", "")
        pow_cal = pred.get("pow_zabudowy_calosc", "")
        budynki = pred.get("budynki") or []
        media   = pred.get("media") or []

        n_rows = max(len(budynki), len(media), 1)
        for i in range(n_rows):
            b = budynki[i] if i < len(budynki) else {}
            m = media[i]   if i < len(media)   else ""

            ws.append([
                nr           if i == 0 else "",
                sposob       if i == 0 else "",
                flaga        if i == 0 else "",
                nazwa        if i == 0 else "",
                adres        if i == 0 else "",
                teren        if i == 0 else "",
                pow_cal      if i == 0 else "",
                (f"{b.get('oznaczenie','')}: {b.get('szerokosc_elewacji','')}" if b else ""),
                (f"{b.get('oznaczenie','')}: {b.get('suma_pow_nadziemnych','')}" if b else ""),
                (f"{b.get('oznaczenie','')}: {b.get('suma_pow_podziemnych','')}" if b else ""),
                (f"{b.get('oznaczenie','')}: {b.get('wys_gornej_krawedzi','')}" if b else ""),
                (f"{b.get('oznaczenie','')}: {b.get('wysokosc_zabudowy','')}" if b else ""),
                (f"{b.get('oznaczenie','')}: {b.get('ilosc_kond_nadziemnych','')}" if b else ""),
                (f"{b.get('oznaczenie','')}: {b.get('ilosc_kond_podziemnych','')}" if b else ""),
                (f"{b.get('oznaczenie','')}: {b.get('geometria_dachu','')}" if b else ""),
                m,
            ])

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = min(
            max(len(str(c.value or "")) for c in col) + 2, 60
        )
    wb.save(str(output_path))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Serve the QA VLM extraction API")
    ap.add_argument("--model",               default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--lora-adapter",        default="")
    ap.add_argument("--port",                type=int, default=8081)
    ap.add_argument("--max-pixels",          type=int, default=501_760)
    ap.add_argument("--max-pages",           type=int, default=6)
    ap.add_argument("--preprocess",          action="store_true",
                    help="Enable adaptive preprocessing (margin crop + blank-page skip)")
    ap.add_argument("--preprocess-debug-dir", default="",
                    help="Save preprocessed page images here for inspection "
                         "(e.g. finetune_qwen_vl_qa/data/preprocess_debug). "
                         "One subfolder per PDF, plus preprocess_stats.json.")
    args = ap.parse_args()

    _max_pages            = args.max_pages
    _preprocess           = args.preprocess
    _preprocess_debug_dir = args.preprocess_debug_dir

    if _preprocess:
        print(f"Adaptive preprocessing: ENABLED (blank threshold=0.15)")
        if _preprocess_debug_dir:
            print(f"  Debug images → {_preprocess_debug_dir}")
    load_model(args.model, args.lora_adapter, args.max_pixels)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
