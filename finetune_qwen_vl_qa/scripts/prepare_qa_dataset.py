from __future__ import annotations

"""
Build the QA-pair training dataset by:
  1. Reading the parsed labels.json (produced by parse_excel_labels.py).
  2. Scanning the PDF directory.
  3. Matching each PDF to a label by extracting the permit number from the filename.
  4. Rendering each PDF to page images (stored persistently in data/images/).
  5. Writing train.jsonl and val.jsonl, one record per DOCUMENT (not per page).

JSONL record format:
  {
    "pdf_stem":     "wz_808",
    "nr_wniosku":   "808",
    "image_paths":  ["/abs/path/data/images/wz_808_p001.png", ...],
    "prompt":       "<extraction prompt>",
    "target_json":  { ... }   ← the structured record from labels.json
  }

PDF filename → nr_wniosku matching rules:
  - "wz_808.pdf"          → 808   (strip "wz_" prefix, parse int)
  - "935 wniosek.pdf"     → 935   (extract leading digits)
  - "wz_un_001.pdf"       → None  (alpha-numeric id, no Excel match expected)
  - "wz_zal_001.pdf"      → None  (attachment, no record)
  - "940 wniosek- SAMA PANI MAGDA.pdf" → 940

Usage:
    source finetune_qwen_vl_pytorch/.venv311/bin/activate

    python finetune_qwen_vl_qa/scripts/prepare_qa_dataset.py \
        --pdf-dir        "Project Files" \
        --labels-json    finetune_qwen_vl_qa/data/labels.json \
        --output-dir     finetune_qwen_vl_qa/data \
        [--dpi           200] \
        [--max-pages     6] \
        [--val-split     0.15] \
        [--seed          42]

Unlabeled PDFs (no matching Excel row) are still rendered and included in a
separate unlabeled.jsonl so you can add labels later or use them for zero-shot
testing.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from qa_utils import build_qa_prompt, render_pdf_pages


# ---------------------------------------------------------------------------
# PDF → nr_wniosku matching
# ---------------------------------------------------------------------------

def extract_nr(pdf_stem: str) -> str | None:
    """
    Try to extract a numeric permit number from a PDF filename stem.

    Supported patterns:
      - "wz_808"          → "808"
      - "935 wniosek"     → "935"
      - "940 wniosek- …"  → "940"
      - "wz_un_001"       → None  (non-numeric suffix)
      - "wz_zal_001"      → None
    """
    stem = pdf_stem.strip()

    # Pattern 1: "wz_<digits>"  (e.g. wz_808, wz_945)
    m = re.fullmatch(r"wz_(\d+)", stem)
    if m:
        return str(int(m.group(1)))

    # Pattern 2: leading digits before a space  (e.g. "935 wniosek", "940 wniosek- ...")
    m = re.match(r"^(\d+)\b", stem)
    if m:
        return str(int(m.group(1)))

    # Pattern 3: trailing digits after last underscore (wz_un_001 → 1, but likely no Excel match)
    # We return it as a candidate; the caller decides if it's in labels.
    m = re.search(r"_(\d+)$", stem)
    if m:
        return str(int(m.group(1)))

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build QA-pair JSONL dataset from PDFs + Excel labels."
    )
    ap.add_argument("--pdf-dir",     required=True, help="Folder containing source PDFs")
    ap.add_argument("--labels-json", required=True, help="labels.json from parse_excel_labels.py")
    ap.add_argument("--output-dir",  required=True, help="Output root (images/ + *.jsonl go here)")
    ap.add_argument("--dpi",         type=int, default=200,
                    help="Render DPI per page (default 200; lower = fewer image tokens)")
    ap.add_argument("--max-pages",   type=int, default=6,
                    help="Max pages to render per PDF (default 6). "
                         "Key fields for WZ forms are on pages 1-4.")
    ap.add_argument("--val-split",   type=float, default=0.15)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    pdf_dir   = Path(args.pdf_dir).expanduser().resolve()
    labels_js = Path(args.labels_json).expanduser().resolve()
    out_dir   = Path(args.output_dir).expanduser().resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, dict] = json.loads(labels_js.read_text(encoding="utf-8"))
    print(f"Loaded {len(labels)} label(s) from {labels_js}")

    prompt = build_qa_prompt()

    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDF(s)")

    # Split by document before rendering (no data leakage)
    shuffled = pdfs.copy()
    random.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_split))
    val_set = {p.name for p in shuffled[:n_val]}
    print(f"Val: {n_val}  Train: {len(pdfs) - n_val}")

    train_labeled: list[dict] = []
    val_labeled:   list[dict] = []
    unlabeled:     list[dict] = []

    for pdf in tqdm(pdfs, desc="Rendering PDFs"):
        try:
            page_paths = render_pdf_pages(pdf, images_dir, dpi=args.dpi, max_pages=args.max_pages)
        except Exception as exc:
            print(f"  SKIP {pdf.name}: {exc}")
            continue

        nr = extract_nr(pdf.stem)
        label = labels.get(nr) if nr else None

        record: dict = {
            "pdf_stem":    pdf.stem,
            "nr_wniosku":  nr or "",
            "image_paths": [str(p) for p in page_paths],
            "prompt":      prompt,
            "target_json": label or {},
        }

        if not label:
            unlabeled.append(record)
        elif pdf.name in val_set:
            val_labeled.append(record)
        else:
            train_labeled.append(record)

    write_jsonl(train_labeled, out_dir / "train.jsonl")
    write_jsonl(val_labeled,   out_dir / "val.jsonl")
    write_jsonl(unlabeled,     out_dir / "unlabeled.jsonl")

    print(f"\nDataset ready in: {out_dir}")
    print(f"  train.jsonl:     {len(train_labeled)} labeled documents")
    print(f"  val.jsonl:       {len(val_labeled)} labeled documents")
    print(f"  unlabeled.jsonl: {len(unlabeled)} documents without Excel label")

    if unlabeled:
        print("\nUnlabeled PDFs (no matching Excel row):")
        for r in unlabeled[:20]:
            print(f"  {r['pdf_stem']} (extracted nr: {r['nr_wniosku'] or 'n/a'})")
        if len(unlabeled) > 20:
            print(f"  ... and {len(unlabeled) - 20} more")
        print("\nTo add labels: find the matching row in the Excel and add to labels.json,")
        print("then re-run this script (already-rendered images are reused).")


if __name__ == "__main__":
    main()
