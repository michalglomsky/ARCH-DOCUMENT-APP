from __future__ import annotations

"""
Local VLM HTTP server — loads Qwen2.5-VL + optional LoRA adapter and serves
the extraction endpoint expected by demo_app's LocalVLMProvider.

API contract (matches demo_app/extractor_demo/providers.py):
  POST /extract
  Request:  {"pdf_path": "/abs/path/to/file.pdf", "schema_fields": ["f1", "f2", ...]}
  Response: {"f1": "...", "f2": "...", ..., "needs_review": true/false,
             "_pages_processed": <N>}

  GET /health  → {"status": "ok", "device": "mps"}

Usage:
    source finetune_qwen_vl_pytorch/.venv311/bin/activate

    # Zero-shot (base model only):
    python finetune_qwen_vl_pytorch/scripts/serve_local_vlm.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct

    # With fine-tuned LoRA adapter:
    python finetune_qwen_vl_pytorch/scripts/serve_local_vlm.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --lora-adapter finetune_qwen_vl_pytorch/out/lora_run1

Wait for "Model ready" before starting demo_app.
Loading Qwen2.5-VL-7B typically takes 1–3 minutes on first download.
"""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from vlm_utils import (
    build_prompt,
    merge_page_results,
    parse_json_response,
    render_pdf_to_images,
    run_inference,
)


# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------

class _State:
    model: Any = None
    processor: Any = None
    device: str = "cpu"
    dpi: int = 250
    max_new_tokens: int = 512


_state = _State()
_inference_lock: asyncio.Lock | None = None  # initialised inside lifespan

app = FastAPI(title="Local VLM Extraction Server", version="1.0")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    global _inference_lock
    _inference_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "device": _state.device}


class ExtractRequest(BaseModel):
    pdf_path: str
    schema_fields: list[str]


@app.post("/extract")
async def extract(req: ExtractRequest) -> dict[str, Any]:
    pdf_path = Path(req.pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")

    prompt = build_prompt(req.schema_fields)

    # Serialise inference — PyTorch is not thread-safe under concurrent requests
    assert _inference_lock is not None
    async with _inference_lock:
        tmp_dir, page_images = render_pdf_to_images(pdf_path, dpi=_state.dpi)
        try:
            page_results: list[dict[str, Any]] = []
            for img_path in page_images:
                raw = run_inference(
                    _state.model,
                    _state.processor,
                    img_path,
                    prompt,
                    device=_state.device,
                    max_new_tokens=_state.max_new_tokens,
                )
                page_results.append(parse_json_response(raw))
                if _state.device == "mps":
                    torch.mps.empty_cache()
        finally:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

    merged = merge_page_results(page_results, req.schema_fields)
    merged["_pages_processed"] = len(page_results)
    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Serve Qwen2.5-VL as a local extraction API")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="HuggingFace model ID or local path")
    ap.add_argument("--lora-adapter", default="",
                    help="Path to saved LoRA adapter directory (optional)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dpi", type=int, default=250, help="PDF render DPI")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    _state.device = "mps" if torch.backends.mps.is_available() else "cpu"
    _state.dpi = args.dpi
    _state.max_new_tokens = args.max_new_tokens

    print(f"Loading processor from: {args.model}")
    _state.processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=1_003_520
    )

    print(f"Loading model from: {args.model}  (device={_state.device})")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=None,
    )
    base_model.to(_state.device)

    if args.lora_adapter:
        print(f"Loading LoRA adapter: {args.lora_adapter}")
        base_model = PeftModel.from_pretrained(base_model, args.lora_adapter)
        # Merge weights for faster inference (no adapter overhead per token)
        base_model = base_model.merge_and_unload()
        print("LoRA adapter merged into base model.")

    base_model.eval()
    _state.model = base_model

    print(f"\nModel ready on {_state.device}.")
    print(f"Serving on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
