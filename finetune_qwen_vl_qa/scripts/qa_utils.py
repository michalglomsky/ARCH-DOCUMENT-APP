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
import numpy as np
import torch
from PIL import Image, ImageEnhance
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
- Fields often appear in multiple locations across pages (headers, tables, stamps, cover letters, attachments). Scan ALL pages for each field and use the most legible occurrence to determine the correct value.
- If the same field has conflicting values in different locations, prefer printed or typed text over handwriting, and set needs_review to true.
- Output ONLY the JSON object — no explanation, no markdown code fences, no extra text."""


def build_qa_prompt() -> str:
    return _QA_PROMPT


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def render_pdf_pages(
    pdf_path: Path,
    images_dir: Path,
    dpi: int = 250,
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


def render_pdf_to_tmp(pdf_path: Path, dpi: int = 250, max_pages: int = 8) -> tuple[Path, list[Path]]:
    """
    Render PDF pages to a temporary directory.
    Returns (tmp_dir, page_paths). Caller must delete tmp_dir when done.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="qa_pages_"))
    paths = render_pdf_pages(pdf_path, tmp_dir, dpi=dpi, max_pages=max_pages)
    return tmp_dir, paths


# ---------------------------------------------------------------------------
# Adaptive preprocessing  (based on: Input-Adaptive Visual Preprocessing
#   for Efficient Fast Vision-Language Model Inference, Cahyani et al.)
# ---------------------------------------------------------------------------

# Complexity score below this → page treated as blank/cover and skipped.
# Set to 0.0 to disable blank-page skipping entirely (all pages are kept).
BLANK_THRESHOLD: float = 0.0

# White-pixel threshold for margin detection (0-255).
# Pixels at or above this value are treated as background.
WHITE_THRESHOLD: int = 240

# Fractional padding added around cropped content (relative to page dimensions).
CROP_PADDING: float = 0.02

# Contrast enhancement factor applied after grayscale conversion.
# 1.0 = no change, 1.3 = mild boost (improves handwriting / stamp legibility).
CONTRAST_FACTOR: float = 1.3


def _compute_complexity(gray: np.ndarray) -> float:
    """
    Compute a visual complexity score in [0, 1] for a grayscale image array.

    Combines two lightweight signals:
      - Edge density  : mean absolute gradient magnitude, normalised to [0, 1].
                        High on pages with dense text, tables, stamps.
      - Entropy       : Shannon entropy of the pixel histogram, normalised by
                        log2(256) = 8.  High when many distinct intensity levels
                        are present (mixed text + background + stamps).

    The two scores are averaged with equal weight.
    """
    # Gradient magnitude via first-order differences (fast, no cv2 needed)
    dx = np.abs(np.diff(gray.astype(np.float32), axis=1))
    dy = np.abs(np.diff(gray.astype(np.float32), axis=0))
    edge_score = min(float(dx.mean() + dy.mean()) / 2.0 / 30.0, 1.0)

    # Shannon entropy of the 256-bin histogram, normalised to [0, 1]
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64) / hist.sum()
    mask = hist > 0
    entropy = float(-np.sum(hist[mask] * np.log2(hist[mask])))
    entropy_score = min(entropy / 8.0, 1.0)

    return float(0.5 * edge_score + 0.5 * entropy_score)


def _crop_margins(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Crop white/near-white margins from a PIL image.

    Algorithm:
      1. Convert to grayscale.
      2. Find all pixels darker than WHITE_THRESHOLD (content pixels).
      3. Compute the bounding box of those pixels.
      4. Expand it by CROP_PADDING on each side.
      5. Return the cropped image and the crop box (left, top, right, bottom).

    If less than 0.5 % of pixels are content (essentially blank), the original
    image is returned unchanged so the blank-page guard in preprocess_pages()
    can handle it cleanly.
    If cropping would remove less than 2 % of the area, the image is also
    returned unchanged (not worth the overhead).
    """
    gray = np.array(img.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    content = gray < WHITE_THRESHOLD

    # Guard: almost blank — return original unchanged
    if content.sum() < 0.005 * h * w:
        return img, (0, 0, w, h)

    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])

    # Add padding
    pad_h = max(1, int(h * CROP_PADDING))
    pad_w = max(1, int(w * CROP_PADDING))
    top    = max(0, top    - pad_h)
    bottom = min(h, bottom + pad_h)
    left   = max(0, left   - pad_w)
    right  = min(w, right  + pad_w)

    crop_box = (left, top, right, bottom)

    # Skip crop if it removes less than 2 % of the area (not worth it)
    area_ratio = ((right - left) * (bottom - top)) / (w * h)
    if area_ratio > 0.98:
        return img, (0, 0, w, h)

    return img.crop(crop_box), crop_box


def preprocess_pages(
    page_paths: list[Path],
    out_dir: Path | None = None,
    blank_threshold: float = BLANK_THRESHOLD,
    enabled: bool = True,
) -> tuple[list[Path], list[dict]]:
    """
    Apply adaptive preprocessing to rendered PDF page images.

    Per page:
      1. Compute visual complexity score.
      2. If score < blank_threshold → skip (blank / cover page).
      3. Otherwise → crop white margins.
      4. Convert to grayscale and enhance contrast (CONTRAST_FACTOR).
      5. Save result to out_dir (or overwrite in-place if out_dir is None).

    Parameters
    ----------
    page_paths      : Rendered PNG paths to process.
    out_dir         : Where to write processed images.
                      None  → overwrite the originals in-place.
                      Path  → save alongside originals (originals are preserved).
    blank_threshold : Complexity below this is treated as a blank page (default 0.15).
    enabled         : Set False to return page_paths unchanged — easy on/off toggle.

    Returns
    -------
    processed_paths : Pages that survived preprocessing (skipped pages excluded).
    stats           : List of per-page dicts with complexity, decision, sizes, crop_box.
                      Write this to JSON to inspect what the preprocessor decided.
    """
    if not enabled:
        return list(page_paths), []

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    processed: list[Path] = []
    stats: list[dict] = []

    for p in page_paths:
        img  = Image.open(str(p)).convert("RGB")
        gray = np.array(img.convert("L"), dtype=np.uint8)
        complexity = _compute_complexity(gray)

        stat: dict = {
            "file":          p.name,
            "complexity":    round(complexity, 4),
            "original_size": list(img.size),   # [w, h]
        }

        if complexity < blank_threshold:
            stat["decision"]        = "skipped"
            stat["reason"]          = f"complexity {complexity:.3f} < threshold {blank_threshold}"
            stat["processed_size"]  = None
            stat["crop_box"]        = None
            stats.append(stat)
            continue   # page excluded from output

        cropped, crop_box = _crop_margins(img)
        was_cropped = crop_box != (0, 0, img.size[0], img.size[1])
        stat["decision"]       = "cropped" if was_cropped else "kept"
        stat["crop_box"]       = list(crop_box)       # [left, top, right, bottom]

        # Grayscale + contrast enhancement
        cropped = cropped.convert("L")
        cropped = ImageEnhance.Contrast(cropped).enhance(CONTRAST_FACTOR)

        stat["processed_size"] = list(cropped.size)   # [w, h]

        out_path = (out_dir / p.name) if out_dir is not None else p
        cropped.save(str(out_path))
        stat["out_path"] = str(out_path)

        stats.append(stat)
        processed.append(out_path)

    return processed, stats


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
