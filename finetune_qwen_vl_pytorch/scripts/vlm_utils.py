from __future__ import annotations

"""
Shared inference utilities for Qwen2.5-VL extraction scripts.

Used by: prepare_dataset.py, serve_local_vlm.py, evaluate.py
"""

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import torch
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a document extraction assistant for Polish construction permit application forms (wnioski o pozwolenie na budowę).

Extract the following fields from the form image.
Return ONLY a valid JSON object with exactly these keys: {fields_json}

Rules:
- Copy text exactly as printed or handwritten — preserve Polish characters (ą ę ó ś ź ż ć ń ł).
- Use "[NIECZYTELNE]" for any text that cannot be clearly read.
- Use "[BRAK]" for fields that are not present on this page or form.
- Set "needs_review" to true if ANY field is uncertain, illegible, ambiguous, or contains [NIECZYTELNE].
- Set "needs_review" to false only when every field was read with confidence.
- Output ONLY the JSON object — no explanation, no markdown code fences, no extra text."""


def build_prompt(schema_fields: list[str]) -> str:
    """Build the extraction prompt for the given schema field list."""
    non_review = [f for f in schema_fields if f != "needs_review"]
    fields_json = json.dumps(non_review + ["needs_review"], ensure_ascii=False)
    return _PROMPT_TEMPLATE.format(fields_json=fields_json)


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def render_pdf_to_images(pdf_path: str | Path, dpi: int = 250) -> tuple[Path, list[Path]]:
    """
    Render every page of a PDF to a PNG in a temporary directory.

    Returns (tmp_dir, [page_paths]).
    Caller is responsible for deleting tmp_dir when done:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="vlm_pages_"))
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    paths: list[Path] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out = tmp_dir / f"page_{i + 1:03d}.png"
            pix.save(str(out))
            paths.append(out)
    finally:
        doc.close()
    return tmp_dir, paths


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: Any,
    processor: Any,
    image_path: str | Path,
    prompt: str,
    device: str,
    max_new_tokens: int = 512,
) -> str:
    """Run a single-image inference and return the raw text response."""
    image_path = str(Path(image_path).expanduser().resolve())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=False,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict[str, Any]:
    """
    Parse model output into a dict, tolerating markdown fences and extra text.
    Falls back gracefully: returns {"needs_review": True, "_parse_error": ...}.
    """
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)

    # Try to find a JSON object anywhere in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last attempt: try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"needs_review": True, "_parse_error": text[:300]}


# ---------------------------------------------------------------------------
# Multi-page merging
# ---------------------------------------------------------------------------

def merge_page_results(
    page_results: list[dict[str, Any]],
    schema_fields: list[str],
) -> dict[str, Any]:
    """
    Merge per-page extraction dicts into one document-level result.

    Strategy:
    - For each field: first non-empty, non-[BRAK] value wins.
    - needs_review = True if any page set it True.
    - _parse_error fields are collected and discarded in the final result.
    """
    merged: dict[str, Any] = {}
    for field in schema_fields:
        if field == "needs_review":
            continue
        for result in page_results:
            v = result.get(field, "")
            if v and v not in ("[BRAK]", "", None):
                merged[field] = v
                break
        if field not in merged:
            merged[field] = "[BRAK]"

    merged["needs_review"] = any(r.get("needs_review", False) for r in page_results)
    return merged
