from __future__ import annotations

"""
Shared utilities for the QA-pair VLM pipeline.

Key differences from the original vlm_utils.py:
- Multi-image inference: all pages of a document are sent as a single user message.
- The target is a nested JSON (with "budynki" list and "media" list) rather than
  a flat per-page dict.
- render_pdf_pages saves images to a persistent directory (not a tmpdir) so that
  dataset JSONL records remain valid across runs.
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

_QA_PROMPT = """\
You are a document extraction assistant for Polish construction permit application forms (wnioski o warunki zabudowy / pozwolenie na budowę).

All pages of ONE permit application are shown above.

Extract the data and return a single JSON object with EXACTLY this structure:
{
  "nr_wniosku": "<permit number as string>",
  "sposob_wypelnienia": "<Komputerowo | Ręcznie | Ręcznie (mało czytelne) | Ręcznie (niestarannie) | ...>",
  "flaga_7_9": "<BRAK | TAK | TAK (zaznaczono TAK) | BRAK (zaznaczono NIE) | ...>",
  "nazwa_inwestycji": "<full investment name in UPPERCASE as on the form>",
  "adres": "<address of the investment>",
  "teren_inwestycji": "<plot/parcel description>",
  "pow_zabudowy_calosc": "<total built-up area range, e.g. od 90,0 m2 do 350,0 m2>",
  "budynki": [
    {
      "oznaczenie": "<building label, e.g. 1. Mieszkalny | 2. Gospodarczy | 3. Garaż>",
      "szerokosc_elewacji": "<facade width range>",
      "suma_pow_nadziemnych": "<above-ground area>",
      "suma_pow_podziemnych": "<underground area>",
      "wys_gornej_krawedzi": "<upper edge height>",
      "wysokosc_zabudowy": "<building height>",
      "ilosc_kond_nadziemnych": "<above-ground floors>",
      "ilosc_kond_podziemnych": "<underground floors>",
      "geometria_dachu": "<roof geometry and pitch>"
    }
  ],
  "media": ["<Woda: ...>", "<Prąd: ...>", "<Gaz: ...>", "<Ciepło: ...>", "<Ścieki: ...>"],
  "needs_review": <true | false>
}

Rules:
- Copy text EXACTLY as printed or handwritten — preserve Polish characters (ą ę ó ś ź ż ć ń ł).
- Include one entry in "budynki" per distinct building/structure on the form.
- For forms about infrastructure (tanks, paving, etc.) with no building parameters, use "[Nie dotyczy]" for building fields and set "budynki" to a single object with oznaczenie "[Obiekt inżynierski]".
- Only include media types that are actually present on the form.
- Use "[NIECZYTELNE]" for text that cannot be clearly read.
- Use "[BRAK]" for fields absent from this form.
- Set "needs_review" to true if ANY field is uncertain, illegible, or ambiguous.
- Output ONLY the JSON object — no explanation, no markdown code fences, no extra text."""


def build_qa_prompt() -> str:
    return _QA_PROMPT


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def render_pdf_pages(
    pdf_path: Path,
    images_dir: Path,
    dpi: int = 200,
    max_pages: int = 8,
) -> list[Path]:
    """
    Render up to max_pages pages of a PDF to PNG files in images_dir.
    File names: <pdf_stem>_p001.png, <pdf_stem>_p002.png, ...

    Lower default DPI (200 vs 250) because we pass multiple images per call.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    paths: list[Path] = []
    try:
        n_pages = min(doc.page_count, max_pages)
        for i in range(n_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out = images_dir / f"{pdf_path.stem}_p{i + 1:03d}.png"
            pix.save(str(out))
            paths.append(out)
    finally:
        doc.close()
    return paths


def render_pdf_to_tmp(pdf_path: Path, dpi: int = 200, max_pages: int = 8) -> tuple[Path, list[Path]]:
    """
    Render PDF pages to a temporary directory.
    Returns (tmp_dir, page_paths). Caller must delete tmp_dir when done.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="qa_pages_"))
    paths = render_pdf_pages(pdf_path, tmp_dir, dpi=dpi, max_pages=max_pages)
    return tmp_dir, paths


# ---------------------------------------------------------------------------
# Multi-image inference
# ---------------------------------------------------------------------------

def build_messages(image_paths: list[Path], prompt: str) -> list[dict[str, Any]]:
    """Build a chat message list with all page images + the extraction prompt."""
    content: list[dict[str, Any]] = []
    for img in image_paths:
        abs_path = str(img.expanduser().resolve())
        content.append({"type": "image", "image": f"file://{abs_path}"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_inference_multipage(
    model: Any,
    processor: Any,
    image_paths: list[Path],
    prompt: str,
    device: str,
    max_new_tokens: int = 1024,
) -> str:
    """Run multi-image inference and return the raw text response."""
    messages = build_messages(image_paths, prompt)
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
    Parse model output into a dict.
    Tolerates markdown fences and surrounding text.
    Falls back gracefully: returns {"needs_review": True, "_parse_error": ...}.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"needs_review": True, "_parse_error": text[:300]}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

_FLAT_FIELDS = [
    "nr_wniosku", "sposob_wypelnienia", "flaga_7_9",
    "nazwa_inwestycji", "adres", "teren_inwestycji", "pow_zabudowy_calosc",
]
_BUILDING_FIELDS = [
    "szerokosc_elewacji", "suma_pow_nadziemnych", "suma_pow_podziemnych",
    "wys_gornej_krawedzi", "wysokosc_zabudowy",
    "ilosc_kond_nadziemnych", "ilosc_kond_podziemnych", "geometria_dachu",
]


def _normalise(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def score_prediction(pred: dict, gold: dict) -> dict[str, Any]:
    """
    Compare a predicted JSON against the gold label.
    Returns a per-field accuracy dict plus a summary.
    """
    results: dict[str, bool] = {}

    # --- flat fields ---
    for field in _FLAT_FIELDS:
        results[field] = _normalise(pred.get(field)) == _normalise(gold.get(field))

    # --- buildings: match by index (order matters) ---
    pred_blds = pred.get("budynki", []) or []
    gold_blds = gold.get("budynki", []) or []
    n_bld = max(len(pred_blds), len(gold_blds))
    for i in range(n_bld):
        pb = pred_blds[i] if i < len(pred_blds) else {}
        gb = gold_blds[i] if i < len(gold_blds) else {}
        for field in _BUILDING_FIELDS:
            key = f"bld{i+1}_{field}"
            results[key] = _normalise(pb.get(field)) == _normalise(gb.get(field))

    # --- media: compare as sorted sets ---
    pred_media = set(_normalise(m) for m in (pred.get("media") or []))
    gold_media = set(_normalise(m) for m in (gold.get("media") or []))
    results["media"] = pred_media == gold_media

    total = len(results)
    correct = sum(results.values())
    return {
        "field_scores": results,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }
