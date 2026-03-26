#!/usr/bin/env python3
"""
build_dataset.py — Build fine-tuning JSONL dataset from PDFs + Excel labels.

Renders PDF pages to PNG, pairs each document with its Excel ground-truth label,
validates the JSON, stratifies by document difficulty, and writes three JSONL splits.

Outputs
-------
  <out_dir>/images/          Pre-rendered PNG pages for all matched PDFs
  <out_dir>/train.jsonl      80 % — 97 records
  <out_dir>/val.jsonl        10 % — 12 records
  <out_dir>/test.jsonl       10 % — 12 records
  <out_dir>/skipped.txt      PDFs with no Excel label (manual review)
  <out_dir>/dataset_stats.txt Human-readable summary

Usage
-----
  python finetune_qwen_vl_qa/scripts/build_dataset.py \\
      --excel   /Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup/1-2026-DANE.xlsx \\
      --pdf-dir /Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup \\
      --out-dir finetune_qwen_vl_qa/data

Options
-------
  --dpi         Render resolution in DPI (default: 200 — must match inference DPI)
  --max-pages   Maximum pages per document (default: 6)
  --seed        Random seed for reproducible splits (default: 42)
  --dry-run     Parse and match without rendering or writing files
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or scripts/ directory
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))

from qa_utils import build_qa_prompt, preprocess_pages, render_pdf_pages  # noqa: E402
from excel_utils import parse_labels                                       # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "nr_wniosku", "sposob_wypelnienia", "flaga_7_9",
    "nazwa_inwestycji", "adres", "teren_inwestycji",
    "pow_zabudowy_calosc", "budynki", "media", "needs_review",
}

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


# ---------------------------------------------------------------------------
# Label → clean JSON
# ---------------------------------------------------------------------------

def _contains_nieczytelne(obj: Any) -> bool:
    """Recursively check if any string value contains [NIECZYTELNE]."""
    if isinstance(obj, str):
        return "[NIECZYTELNE]" in obj
    if isinstance(obj, dict):
        return any(_contains_nieczytelne(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_nieczytelne(v) for v in obj)
    return False


def label_to_assistant_string(label: dict) -> str:
    """
    Convert a parsed Excel label dict to the assistant JSON string used in training.

    - Adds 'needs_review': true if any field contains [NIECZYTELNE]
    - Strips internal-only keys (those starting with '_')
    - Serialises to compact JSON (no whitespace) — one token sequence per field
    """
    record = {k: v for k, v in label.items() if not k.startswith("_")}

    # Auto-set needs_review based on illegible fields
    if "needs_review" not in record:
        record["needs_review"] = _contains_nieczytelne(record)

    # Ensure key order matches the prompt schema for consistency
    ordered = {
        "nr_wniosku":          record.get("nr_wniosku", ""),
        "sposob_wypelnienia":  record.get("sposob_wypelnienia", ""),
        "flaga_7_9":           record.get("flaga_7_9", ""),
        "nazwa_inwestycji":    record.get("nazwa_inwestycji", ""),
        "adres":               record.get("adres", ""),
        "teren_inwestycji":    record.get("teren_inwestycji", ""),
        "pow_zabudowy_calosc": record.get("pow_zabudowy_calosc", ""),
        "budynki":             record.get("budynki", []),
        "media":               record.get("media", []),
        "needs_review":        bool(record.get("needs_review", False)),
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def validate_assistant_string(s: str, doc_id: str) -> list[str]:
    """
    Parse the assistant string and return a list of validation warnings.
    Empty list means the record is clean.
    """
    warnings = []
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        return [f"JSON parse error: {e}"]

    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        warnings.append(f"Missing keys: {sorted(missing)}")

    if not isinstance(obj.get("budynki"), list):
        warnings.append("'budynki' is not a list")
    elif len(obj["budynki"]) == 0:
        warnings.append("'budynki' is empty — expected at least one entry")

    if not isinstance(obj.get("media"), list):
        warnings.append("'media' is not a list")

    return warnings


# ---------------------------------------------------------------------------
# PDF → nr_wniosku matching
# ---------------------------------------------------------------------------

def extract_nr_from_stem(stem: str) -> str | None:
    """
    Extract the permit number from a PDF stem.

    Examples:
      wz_808        → "808"
      pozw_1024_A   → "1024"
      00312         → "312"
    """
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return None
    # Use the last (most specific) numeric group
    try:
        return str(int(numbers[-1]))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def assign_stratum(label: dict) -> str:
    """
    Assign a stratum label for stratified splitting.

    Two axes:
      - n_buildings: single (1) vs multi (2+)
      - handwritten: Komputerowo vs anything else (Ręcznie variants)

    Produces four buckets: single_typed, single_hand, multi_typed, multi_hand
    """
    n_bld = len(label.get("budynki") or [])
    bld_cat = "multi" if n_bld > 1 else "single"

    sposob = (label.get("sposob_wypelnienia") or "").lower()
    hand_cat = "typed" if "komputerowo" in sposob else "hand"

    return f"{bld_cat}_{hand_cat}"


def stratified_split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    """
    Split records into train/val/test while preserving stratum proportions.

    Algorithm:
      1. Group by stratum
      2. Within each stratum, shuffle and allocate ~10 % to val, ~10 % to test
      3. Remainder goes to train
    """
    rng = random.Random(seed)

    buckets: dict[str, list[dict]] = {}
    for r in records:
        s = r["stratum"]
        buckets.setdefault(s, []).append(r)

    train, val, test = [], [], []

    for stratum, items in sorted(buckets.items()):
        rng.shuffle(items)
        n = len(items)
        n_val  = max(1, round(n * SPLIT_RATIOS["val"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test"]))
        # Guard: don't over-allocate for tiny strata
        n_val  = min(n_val,  max(0, n - 2))
        n_test = min(n_test, max(0, n - n_val - 1))

        test  += items[:n_test]
        val   += items[n_test: n_test + n_val]
        train += items[n_test + n_val:]

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# JSONL record builder
# ---------------------------------------------------------------------------

def build_record(
    doc_id: str,
    image_paths: list[Path],
    assistant_string: str,
    prompt: str,
) -> dict:
    """Build one JSONL record from rendered page paths and the assistant JSON."""
    content: list[dict] = []
    for img in image_paths:
        content.append({"type": "image", "image": f"file://{img.resolve()}"})
    content.append({"type": "text", "text": prompt})

    return {
        "id": doc_id,
        "messages": [
            {"role": "user",      "content": content},
            {"role": "assistant", "content": assistant_string},
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _render_and_preprocess(
    pdfs: list[Path],
    images_dir: Path,
    images_pre_dir: Path,
    dpi: int,
    max_pages: int,
    preprocess: bool,
) -> tuple[dict[str, list[Path]], list[tuple[str, str]]]:
    """
    Render all PDFs and optionally preprocess them.
    Returns (page_paths_by_stem, render_errors).
    """
    page_paths_by_stem: dict[str, list[Path]] = {}
    render_errors: list[tuple[str, str]] = []
    total = len(pdfs)

    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i:3d}/{total}] Rendering {pdf.name} …", end=" ")
        try:
            page_paths = render_pdf_pages(pdf, images_dir, dpi=dpi, max_pages=max_pages)
            print(f"{len(page_paths)} pages", end="")
        except Exception as e:
            print(f"  ERROR: {e}")
            render_errors.append((pdf.name, str(e)))
            continue

        if not page_paths:
            render_errors.append((pdf.name, "rendered 0 pages"))
            print()
            continue

        if preprocess:
            pre_out = images_pre_dir / pdf.stem
            page_paths, pp_stats = preprocess_pages(page_paths, out_dir=pre_out, enabled=True)
            pre_out.mkdir(parents=True, exist_ok=True)
            (pre_out / "preprocess_stats.json").write_text(
                json.dumps(pp_stats, indent=2, ensure_ascii=False)
            )
            skipped_p = sum(1 for s in pp_stats if s["decision"] == "skipped")
            cropped_p = sum(1 for s in pp_stats if s["decision"] == "cropped")
            print(f"  →  preprocessed: {len(page_paths)} kept, "
                  f"{skipped_p} skipped, {cropped_p} cropped", end="")
            if not page_paths:
                render_errors.append((pdf.name, "all pages blank after preprocessing"))
                print()
                continue

        print()
        page_paths_by_stem[pdf.stem] = page_paths

    return page_paths_by_stem, render_errors


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render + preprocess PDF pages, and optionally build fine-tuning JSONL splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  Render only (no Excel needed):
    build_dataset.py --pdf-dir /path/to/pdfs --out-dir data --preprocess

  Full dataset build (requires Excel labels):
    build_dataset.py --pdf-dir /path/to/pdfs --excel /path/to/labels.xlsx --out-dir data
""",
    )
    ap.add_argument("--pdf-dir",   required=True, help="Directory containing PDFs")
    ap.add_argument("--out-dir",   required=True, help="Output directory")
    ap.add_argument("--excel",     default="",    help="Path to labels Excel (optional — omit to render images only)")
    ap.add_argument("--dpi",       type=int, default=250, help="Render DPI (default: 250)")
    ap.add_argument("--max-pages", type=int, default=6,   help="Max pages per document (default: 6)")
    ap.add_argument("--seed",      type=int, default=42,  help="Random seed (default: 42)")
    ap.add_argument("--dry-run",    action="store_true",  help="Validate without rendering or writing")
    ap.add_argument("--preprocess", action="store_true",
                    help="Apply adaptive preprocessing (margin crop + blank-page skip). "
                         "Saves processed images to <out_dir>/images_preprocessed/ alongside "
                         "the originals in <out_dir>/images/ so you can compare them directly.")
    args = ap.parse_args()

    pdf_dir        = Path(args.pdf_dir).expanduser().resolve()
    out_dir        = Path(args.out_dir).expanduser().resolve()
    images_dir     = out_dir / "images"
    images_pre_dir = out_dir / "images_preprocessed"
    labels_mode    = bool(args.excel)   # False = render-only, True = full dataset build

    if not pdf_dir.exists():
        sys.exit(f"ERROR: PDF directory not found: {pdf_dir}")

    if labels_mode:
        excel_path = Path(args.excel).expanduser().resolve()
        if not excel_path.exists():
            sys.exit(f"ERROR: Excel file not found: {excel_path}")
    else:
        print("No --excel provided — running in render-only mode (images only, no JSONL).")

    if not args.dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        if args.preprocess:
            images_pre_dir.mkdir(parents=True, exist_ok=True)

    # ---- Find all PDFs ----------------------------------------------------
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"{len(pdfs)} PDFs found in {pdf_dir}")

    # ---- Render-only mode -------------------------------------------------
    if not labels_mode:
        if args.dry_run:
            print("Dry run — would render all PDFs. No files written.")
            return
        page_paths_by_stem, render_errors = _render_and_preprocess(
            pdfs, images_dir, images_pre_dir, args.dpi, args.max_pages, args.preprocess
        )
        print("\n" + "=" * 50)
        print(f"Done.  Rendered {len(page_paths_by_stem)} / {len(pdfs)} PDFs.")
        print(f"  Images (original):     {images_dir}")
        if args.preprocess:
            print(f"  Images (preprocessed): {images_pre_dir}")
        if render_errors:
            print(f"\n  ✗ {len(render_errors)} render errors:")
            for name, err in render_errors:
                print(f"    {name}: {err}")
        print()
        return

    # ---- Full dataset build (Excel mode) ----------------------------------
    print(f"Loading labels from {excel_path} …")
    labels = parse_labels(excel_path)
    print(f"  {len(labels)} labelled records found")

    prompt = build_qa_prompt()
    matched      = []
    skipped      = []
    warnings_log = []

    for pdf in pdfs:
        nr = extract_nr_from_stem(pdf.stem)
        if nr is None:
            skipped.append((pdf.name, "cannot extract nr_wniosku from filename"))
            continue
        label = labels.get(nr)
        if label is None:
            skipped.append((pdf.name, f"no Excel label for nr_wniosku={nr}"))
            continue
        assistant_str = label_to_assistant_string(label)
        issues = validate_assistant_string(assistant_str, pdf.stem)
        if issues:
            warnings_log.append(f"{pdf.name}: {'; '.join(issues)}")
        matched.append({
            "pdf":           pdf,
            "doc_id":        pdf.stem,
            "nr":            nr,
            "label":         label,
            "assistant_str": assistant_str,
            "stratum":       assign_stratum(label),
            "issues":        issues,
        })

    print(f"\nMatched:  {len(matched)} documents")
    print(f"Skipped:  {len(skipped)} documents")
    if warnings_log:
        print(f"Warnings: {len(warnings_log)} records have validation issues (see stats file)")
    if len(matched) == 0:
        sys.exit("ERROR: No matched documents — check PDF filenames contain the permit number.")

    splits = stratified_split(matched, seed=args.seed)
    print(f"\nSplit:  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    for split_name, items in splits.items():
        counts: dict[str, int] = {}
        for item in items:
            counts[item["stratum"]] = counts.get(item["stratum"], 0) + 1
        print(f"  {split_name:6s}: {dict(sorted(counts.items()))}")

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    # Render all matched PDFs
    matched_pdfs = [item["pdf"] for item in matched]
    page_paths_by_stem, render_errors = _render_and_preprocess(
        matched_pdfs, images_dir, images_pre_dir, args.dpi, args.max_pages, args.preprocess
    )

    # Write JSONL splits
    for split_name, items in splits.items():
        jsonl_path = out_dir / f"{split_name}.jsonl"
        records_written = 0
        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in items:
                doc_id = item["doc_id"]
                page_paths = page_paths_by_stem.get(doc_id)
                if not page_paths:
                    continue
                record = build_record(doc_id, page_paths, item["assistant_str"], prompt)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1
        print(f"  → Wrote {records_written} records to {jsonl_path.name}")

    # Write stats
    stats_path = out_dir / "dataset_stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("Dataset generation summary\n" + "=" * 60 + "\n\n")
        f.write(f"Excel file:    {excel_path}\n")
        f.write(f"PDF dir:       {pdf_dir}\n")
        f.write(f"Output dir:    {out_dir}\n")
        f.write(f"Render DPI:    {args.dpi}\n")
        f.write(f"Max pages:     {args.max_pages}\n")
        f.write(f"Random seed:   {args.seed}\n\n")
        f.write(f"Total PDFs:    {len(pdfs)}\n")
        f.write(f"Matched:       {len(matched)}\n")
        f.write(f"Skipped:       {len(skipped)}\n")
        f.write(f"Render errors: {len(render_errors)}\n\n")
        f.write("Splits\n------\n")
        for split_name, items in splits.items():
            f.write(f"  {split_name:6s}: {len(items)}\n")
        f.write("\nStratum breakdown\n-----------------\n")
        for split_name, items in splits.items():
            counts2: dict[str, int] = {}
            for item in items:
                counts2[item["stratum"]] = counts2.get(item["stratum"], 0) + 1
            f.write(f"  {split_name:6s}: {dict(sorted(counts2.items()))}\n")
        if warnings_log:
            f.write(f"\nValidation warnings ({len(warnings_log)})\n" + "-" * 40 + "\n")
            for w in warnings_log:
                f.write(f"  {w}\n")
        if render_errors:
            f.write(f"\nRender errors ({len(render_errors)})\n" + "-" * 40 + "\n")
            for name, err in render_errors:
                f.write(f"  {name}: {err}\n")

    print("\n" + "=" * 50)
    print("Done.")
    print(f"  Images (original):     {images_dir}")
    if args.preprocess:
        print(f"  Images (preprocessed): {images_pre_dir}")
    print(f"  train / val / test:    {out_dir}")
    print(f"  Stats:                 {stats_path}")
    if render_errors:
        print(f"\n  ✗ {len(render_errors)} PDFs failed to render")
    print()


if __name__ == "__main__":
    main()
