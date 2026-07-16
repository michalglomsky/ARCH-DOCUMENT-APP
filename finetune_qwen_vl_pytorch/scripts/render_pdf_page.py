from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF


def main() -> None:
    p = argparse.ArgumentParser(description="Render a single PDF page to an image (PNG).")
    p.add_argument("--pdf", required=True, help="Path to input PDF")
    p.add_argument("--page", type=int, default=1, help="1-based page number")
    p.add_argument("--dpi", type=int, default=250, help="Render DPI")
    p.add_argument("--out", required=True, help="Output PNG path")
    args = p.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        page_index = max(0, args.page - 1)
        if page_index >= doc.page_count:
            raise SystemExit(f"PDF has only {doc.page_count} pages, cannot render page {args.page}.")
        page = doc.load_page(page_index)
        zoom = args.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out_path))
    finally:
        doc.close()

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

