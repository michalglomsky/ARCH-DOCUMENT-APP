from __future__ import annotations

"""
Batch dataset preparation: PDF folder → per-page PNG images + train/val JSONL.

Usage:
    source finetune_qwen_vl_pytorch/.venv311/bin/activate

    python finetune_qwen_vl_pytorch/scripts/prepare_dataset.py \
        --pdf-dir  "Project Files" \
        --output-dir finetune_qwen_vl_pytorch/data \
        [--dpi 250] \
        [--val-split 0.15] \
        [--schema-fields nr_wniosku,adres,...] \
        [--labels-jsonl existing_labels.jsonl]

Output:
    data/images/<pdf_stem>_p<NNN>.png   — rendered page images
    data/train.jsonl                     — training split (labeled or placeholder)
    data/val.jsonl                       — validation split

JSONL record format (one JSON object per line):
    {
      "image_path": "/abs/path/to/image.png",
      "prompt":     "<extraction instruction>",
      "target_json": {}    ← fill this in before training!
    }

Labeling workflow after running this script:
  Option A (manual): open train.jsonl, fill in "target_json" for each record.
  Option B (Nanonets weak labels): run Nanonets on each PDF, export results,
           then pass them via --labels-jsonl to merge automatically.
  Option C (zero-shot bootstrap): run evaluate.py (zero-shot) on val.jsonl,
           manually correct the predictions, save as new labels.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from vlm_utils import build_prompt

# Default schema matching demo_app/config.example.yaml
DEFAULT_SCHEMA_FIELDS = [
    "nr_wniosku",
    "sposob_wypelnienia",
    "flaga_7_9",
    "nazwa_inwestycji",
    "adres",
    "teren_inwestycji",
    "pow_zabudowy_calosc",
    "szerokosc_elewacji",
    "suma_pow_nadziemnych",
    "suma_pow_podziemnych",
    "wys_gornej_krawedzi",
    "wysokosc_zabudowy",
    "ilosc_kond_nadziemnych",
    "ilosc_kond_podziemnych",
    "geometria_dachu",
    "media",
    "needs_review",
]


def render_pdf(pdf_path: Path, images_dir: Path, dpi: int) -> list[Path]:
    """Render all pages of a PDF to PNG files in images_dir."""
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    out_paths: list[Path] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            # Naming: <pdf_stem>_p001.png, <pdf_stem>_p002.png, …
            out = images_dir / f"{pdf_path.stem}_p{i + 1:03d}.png"
            pix.save(str(out))
            out_paths.append(out)
    finally:
        doc.close()
    return out_paths


def load_existing_labels(jsonl_path: Path) -> dict[str, dict]:
    """
    Load an existing labels JSONL and index by image filename stem.
    Supports both:
      - our own JSONL format: {"image_path": "...", "target_json": {...}}
      - simple dict: {"<stem>": {...}} (flat key → value map)
    """
    labels: dict[str, dict] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "image_path" in obj:
            key = Path(obj["image_path"]).stem
            labels[key] = obj.get("target_json", {})
        elif "image_stem" in obj:
            labels[obj["image_stem"]] = obj.get("target_json", {})
    return labels


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render PDFs to images and create training JSONL files."
    )
    ap.add_argument("--pdf-dir", required=True, help="Folder containing source PDFs")
    ap.add_argument("--output-dir", required=True, help="Output root (images/ + *.jsonl written here)")
    ap.add_argument("--dpi", type=int, default=250, help="Render DPI (default 250)")
    ap.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of documents reserved for val set (split by document, not page)",
    )
    ap.add_argument(
        "--schema-fields",
        default="",
        help="Comma-separated field names. Defaults to the built-in WZ schema.",
    )
    ap.add_argument(
        "--labels-jsonl",
        default="",
        help="Optional JSONL with existing 'target_json' values to merge in.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    schema_fields = (
        [f.strip() for f in args.schema_fields.split(",") if f.strip()]
        if args.schema_fields
        else DEFAULT_SCHEMA_FIELDS
    )
    prompt = build_prompt(schema_fields)

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load any existing labels
    existing_labels: dict[str, dict] = {}
    if args.labels_jsonl:
        existing_labels = load_existing_labels(Path(args.labels_jsonl))
        print(f"Loaded {len(existing_labels)} existing label(s) from {args.labels_jsonl}")

    # Discover PDFs
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in: {pdf_dir}")
    print(f"Found {len(pdfs)} PDF(s)")

    # Split by document (not by page) to avoid data leakage
    random.seed(args.seed)
    shuffled = pdfs.copy()
    random.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_split))
    val_set = {p.name for p in shuffled[:n_val]}
    print(f"Val documents: {n_val}  Train documents: {len(pdfs) - n_val}")

    train_records: list[dict] = []
    val_records: list[dict] = []
    n_labeled = 0

    for pdf in tqdm(pdfs, desc="Rendering PDFs"):
        try:
            page_paths = render_pdf(pdf, images_dir, args.dpi)
        except Exception as exc:
            print(f"  SKIP {pdf.name}: {exc}")
            continue

        for img_path in page_paths:
            target = existing_labels.get(img_path.stem, {})
            if target:
                n_labeled += 1
            record = {
                "image_path": str(img_path),
                "prompt": prompt,
                "target_json": target,
            }
            if pdf.name in val_set:
                val_records.append(record)
            else:
                train_records.append(record)

    write_jsonl(train_records, out_dir / "train.jsonl")
    write_jsonl(val_records, out_dir / "val.jsonl")

    print(f"\nDataset ready in: {out_dir}")
    print(f"  Images:        {images_dir}")
    print(f"  train.jsonl:   {len(train_records)} records")
    print(f"  val.jsonl:     {len(val_records)} records")
    print(f"  Labeled so far: {n_labeled} / {len(train_records) + len(val_records)}")

    if n_labeled < len(train_records) + len(val_records):
        unlabeled = len(train_records) + len(val_records) - n_labeled
        print(
            f"\n  ⚠  {unlabeled} records have empty 'target_json'."
            "\n     Fill them in before running train_lora.py."
            "\n     (The trainer skips records with empty target_json.)"
        )


if __name__ == "__main__":
    main()
