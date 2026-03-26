from __future__ import annotations

import argparse
import dataclasses
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import cv2
import fitz  # PyMuPDF
import numpy as np
import yaml
from tqdm import tqdm

try:
    import easyocr
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "easyocr failed to import. Install dependencies with: pip install -r requirements.txt"
    ) from e


CoordSpace = Literal["top_left"]


@dataclass(frozen=True)
class RectTL:
    """
    Rectangle in TOP-LEFT origin coordinate space:
    x grows to the right, y grows downward.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def normalized(self) -> "RectTL":
        return RectTL(
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
        )

    def clamp(self, width: float, height: float) -> "RectTL":
        r = self.normalized()
        return RectTL(
            max(0.0, min(r.x0, width)),
            max(0.0, min(r.y0, height)),
            max(0.0, min(r.x1, width)),
            max(0.0, min(r.y1, height)),
        )

    @property
    def w(self) -> float:
        r = self.normalized()
        return r.x1 - r.x0

    @property
    def h(self) -> float:
        r = self.normalized()
        return r.y1 - r.y0


@dataclass(frozen=True)
class OCRBox:
    text: str
    conf: float
    rect_img: RectTL  # in image pixels, top-left origin


@dataclass(frozen=True)
class AnchorRedactionRect:
    dx: float
    dy: float
    w: float
    h: float
    note: str | None = None


@dataclass(frozen=True)
class AnchorRule:
    anchor_regex: str
    rects: list[AnchorRedactionRect]


@dataclass(frozen=True)
class FixedRule:
    x0: float
    y0: float
    x1: float
    y1: float
    note: str | None = None


@dataclass(frozen=True)
class TextRegexRule:
    """Redact any OCR box whose text matches this pattern directly."""
    pattern: str
    padding: float = 5.0  # extra space (PDF points) around the matched text box
    note: str | None = None


@dataclass(frozen=True)
class PageType:
    name: str
    match_any: list[str]
    anchor_redactions: list[AnchorRule]
    fixed_redactions: list[FixedRule]
    text_redactions: list[TextRegexRule] = dataclasses.field(default_factory=list)


@dataclass(frozen=True)
class Config:
    version: int
    render_dpi: int
    ocr_langs: list[str]
    auto_rotate_ocr_image: bool
    redaction_fill_rgb: tuple[int, int, int]
    output_suffix: str
    page_types: list[PageType]


def _load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    version = int(raw.get("version", 1))
    render_dpi = int(raw.get("render_dpi", 250))
    ocr_langs = list(raw.get("ocr_langs", ["pl"]))
    auto_rotate = bool(raw.get("auto_rotate_ocr_image", False))
    fill = tuple(int(x) for x in raw.get("redaction_fill_rgb", [0, 0, 0]))
    output_suffix = str(raw.get("output_suffix", ".redacted"))

    pts: list[PageType] = []
    for pt in raw.get("page_types", []):
        anchor_rules: list[AnchorRule] = []
        for ar in pt.get("anchor_redactions", []) or []:
            rects = [
                AnchorRedactionRect(
                    dx=float(r["dx"]),
                    dy=float(r["dy"]),
                    w=float(r["w"]),
                    h=float(r["h"]),
                    note=r.get("note"),
                )
                for r in (ar.get("rects", []) or [])
            ]
            anchor_rules.append(AnchorRule(anchor_regex=str(ar["anchor_regex"]), rects=rects))

        fixed_rules: list[FixedRule] = []
        for fr in pt.get("fixed_redactions", []) or []:
            fixed_rules.append(
                FixedRule(
                    x0=float(fr["x0"]),
                    y0=float(fr["y0"]),
                    x1=float(fr["x1"]),
                    y1=float(fr["y1"]),
                    note=fr.get("note"),
                )
            )

        text_rules: list[TextRegexRule] = []
        for tr in pt.get("text_redactions", []) or []:
            text_rules.append(
                TextRegexRule(
                    pattern=str(tr["pattern"]),
                    padding=float(tr.get("padding", 5.0)),
                    note=tr.get("note"),
                )
            )

        pts.append(
            PageType(
                name=str(pt["name"]),
                match_any=[str(p) for p in (pt.get("match_any", []) or [])],
                anchor_redactions=anchor_rules,
                fixed_redactions=fixed_rules,
                text_redactions=text_rules,
            )
        )

    if not pts:
        pts = [PageType(name="fallback_unknown", match_any=[], anchor_redactions=[], fixed_redactions=[])]

    return Config(
        version=version,
        render_dpi=render_dpi,
        ocr_langs=ocr_langs,
        auto_rotate_ocr_image=auto_rotate,
        redaction_fill_rgb=fill,  # type: ignore[assignment]
        output_suffix=output_suffix,
        page_types=pts,
    )


def _render_page_to_image(page: fitz.Page, dpi: int) -> tuple[np.ndarray, float]:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, zoom


def _maybe_autorotate(img_bgr: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return img_bgr
    # Simple heuristic: try 0/90/180/270 and pick the one with the most "horizontal text-like" edges.
    # This is intentionally conservative; you can disable if it harms performance.
    candidates = [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    scores: list[float] = []
    for c in candidates:
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Horizontal projection variance as proxy
        proj = edges.sum(axis=1).astype(np.float32)
        scores.append(float(np.var(proj)))
    return candidates[int(np.argmax(scores))]


def _ocr_page(reader: "easyocr.Reader", img_bgr: np.ndarray) -> list[OCRBox]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb, detail=1, paragraph=False)
    out: list[OCRBox] = []
    for item in results:
        bbox, text, conf = item
        if not text:
            continue
        pts = np.array(bbox, dtype=np.float32)  # 4 points
        x0 = float(np.min(pts[:, 0]))
        y0 = float(np.min(pts[:, 1]))
        x1 = float(np.max(pts[:, 0]))
        y1 = float(np.max(pts[:, 1]))
        out.append(OCRBox(text=str(text), conf=float(conf), rect_img=RectTL(x0, y0, x1, y1).normalized()))
    return out


def _page_text(ocr_boxes: list[OCRBox]) -> str:
    # Join with whitespace; normalize multiple spaces.
    joined = " ".join(b.text for b in ocr_boxes)
    return re.sub(r"\s+", " ", joined).strip()


def _classify_page(cfg: Config, ocr_boxes: list[OCRBox]) -> PageType:
    text = _page_text(ocr_boxes)
    for pt in cfg.page_types:
        if not pt.match_any:
            continue
        for patt in pt.match_any:
            if re.search(patt, text, flags=re.MULTILINE):
                return pt
    # fallback: first page type with empty match_any, else last
    for pt in cfg.page_types:
        if not pt.match_any:
            return pt
    return cfg.page_types[-1]


def _img_rect_to_pdf_rect_tl(
    rect_img: RectTL, page_width_pt: float, page_height_pt: float, img_width_px: int, img_height_px: int
) -> RectTL:
    # Map image px to PDF pt in top-left origin.
    sx = page_width_pt / float(img_width_px)
    sy = page_height_pt / float(img_height_px)
    return RectTL(
        rect_img.x0 * sx,
        rect_img.y0 * sy,
        rect_img.x1 * sx,
        rect_img.y1 * sy,
    ).normalized()


def _rect_tl_to_fitz(rect_tl: RectTL, page_height_pt: float) -> fitz.Rect:
    # Convert TOP-LEFT rect to PyMuPDF rect (origin bottom-left).
    r = rect_tl.normalized()
    x0 = r.x0
    x1 = r.x1
    y0 = page_height_pt - r.y1
    y1 = page_height_pt - r.y0
    return fitz.Rect(x0, y0, x1, y1)


def _collect_redactions_for_page(
    cfg: Config,
    pt: PageType,
    ocr_boxes: list[OCRBox],
    page_width_pt: float,
    page_height_pt: float,
    img_width_px: int,
    img_height_px: int,
) -> tuple[list[RectTL], list[dict]]:
    """
    Returns (redaction_rects, matches) where matches is a list of dicts describing
    what triggered each redaction — useful for preview/audit.
    """
    rects: list[RectTL] = []
    matches: list[dict] = []

    # Fixed rules
    for fr in pt.fixed_redactions:
        rects.append(RectTL(fr.x0, fr.y0, fr.x1, fr.y1).normalized())
        matches.append({"type": "fixed", "note": fr.note or ""})

    # Direct text regex rules (email, phone, etc.)
    for tr in pt.text_redactions:
        try:
            rx = re.compile(tr.pattern)
        except re.error:
            continue
        for box in ocr_boxes:
            if rx.search(box.text):
                pdf_rect = _img_rect_to_pdf_rect_tl(
                    box.rect_img, page_width_pt, page_height_pt, img_width_px, img_height_px
                )
                padded = RectTL(
                    pdf_rect.x0 - tr.padding,
                    pdf_rect.y0 - tr.padding,
                    pdf_rect.x1 + tr.padding,
                    pdf_rect.y1 + tr.padding,
                )
                rects.append(padded)
                matches.append({"type": "text_regex", "pattern": tr.note or tr.pattern, "text": box.text})

    # Anchor rules
    for ar in pt.anchor_redactions:
        try:
            rx = re.compile(ar.anchor_regex)
        except re.error:
            continue

        candidates: list[OCRBox] = [b for b in ocr_boxes if rx.search(b.text)]
        if not candidates:
            continue
        candidates.sort(key=lambda b: (b.conf, b.rect_img.w * b.rect_img.h), reverse=True)
        anchor = candidates[0]

        anchor_pdf = _img_rect_to_pdf_rect_tl(
            anchor.rect_img, page_width_pt, page_height_pt, img_width_px, img_height_px
        )

        for rr in ar.rects:
            rects.append(
                RectTL(
                    anchor_pdf.x0 + rr.dx,
                    anchor_pdf.y0 + rr.dy,
                    anchor_pdf.x0 + rr.dx + rr.w,
                    anchor_pdf.y0 + rr.dy + rr.h,
                ).normalized()
            )
            matches.append({"type": "anchor", "anchor": anchor.text, "note": rr.note or ""})

    # Clamp to page and drop tiny rects
    result = []
    result_matches = []
    for r, m in zip(rects, matches):
        r = r.clamp(page_width_pt, page_height_pt)
        if r.w >= 2 and r.h >= 2:
            result.append(r)
            result_matches.append(m)

    return result, result_matches


def redact_pdf(
    input_pdf: Path,
    output_pdf: Path,
    cfg: Config,
    reader: "easyocr.Reader",
    pages_to_remove: list[int] | None = None,
) -> dict[str, Any]:
    doc = fitz.open(str(input_pdf))
    remove_set = set(pages_to_remove or [])
    stats: dict[str, Any] = {
        "input": str(input_pdf),
        "output": str(output_pdf),
        "pages": [],
        "pages_removed": sorted(remove_set),
    }

    for i in range(doc.page_count):
        if i in remove_set:
            stats["pages"].append({"page_index": i, "removed": True})
            continue

        page = doc.load_page(i)
        page_rect = page.rect
        page_w_pt = float(page_rect.width)
        page_h_pt = float(page_rect.height)

        img_bgr, _zoom = _render_page_to_image(page, cfg.render_dpi)
        img_bgr = _maybe_autorotate(img_bgr, cfg.auto_rotate_ocr_image)

        ocr_boxes = _ocr_page(reader, img_bgr)
        page_type = _classify_page(cfg, ocr_boxes)

        rects_tl, _matches = _collect_redactions_for_page(
            cfg=cfg,
            pt=page_type,
            ocr_boxes=ocr_boxes,
            page_width_pt=page_w_pt,
            page_height_pt=page_h_pt,
            img_width_px=int(img_bgr.shape[1]),
            img_height_px=int(img_bgr.shape[0]),
        )

        # Add redaction annots
        for r in rects_tl:
            fr = _rect_tl_to_fitz(r, page_h_pt)
            page.add_redact_annot(fr, fill=cfg.redaction_fill_rgb)

        # Apply redactions on this page
        if rects_tl:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)

        stats["pages"].append(
            {
                "page_index": i,
                "page_type": page_type.name,
                "ocr_boxes": len(ocr_boxes),
                "redactions": len(rects_tl),
                "matches": _matches,
            }
        )

    # Delete pages in reverse order to preserve indices
    for i in sorted(remove_set, reverse=True):
        if 0 <= i < doc.page_count:
            doc.delete_page(i)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf), deflate=True, garbage=4, clean=True)
    doc.close()
    return stats


def preview_pdf(
    input_pdf: Path,
    cfg: Config,
    reader: "easyocr.Reader",
    show_ocr_boxes: bool = False,
) -> list[dict[str, Any]]:
    """
    Dry-run: detect redaction zones without modifying the PDF.

    Returns a list of per-page dicts:
      - page_index     : int
      - page_type      : str   (classified page type name)
      - redaction_count: int
      - matches        : list  (what triggered each redaction)
      - annotated_bgr  : np.ndarray  (BGR image with boxes drawn)

    When show_ocr_boxes=True, all OCR-detected text boxes are drawn in blue
    so you can see what text was found and calibrate anchor offsets.
    """
    doc = fitz.open(str(input_pdf))
    pages: list[dict[str, Any]] = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        page_rect = page.rect
        page_w_pt = float(page_rect.width)
        page_h_pt = float(page_rect.height)

        img_bgr, _ = _render_page_to_image(page, cfg.render_dpi)
        img_bgr = _maybe_autorotate(img_bgr, cfg.auto_rotate_ocr_image)

        ocr_boxes = _ocr_page(reader, img_bgr)
        page_type = _classify_page(cfg, ocr_boxes)

        rects_tl, matches = _collect_redactions_for_page(
            cfg=cfg,
            pt=page_type,
            ocr_boxes=ocr_boxes,
            page_width_pt=page_w_pt,
            page_height_pt=page_h_pt,
            img_width_px=int(img_bgr.shape[1]),
            img_height_px=int(img_bgr.shape[0]),
        )

        annotated = img_bgr.copy()
        sx = img_bgr.shape[1] / page_w_pt
        sy = img_bgr.shape[0] / page_h_pt

        # Draw OCR boxes in blue (calibration aid)
        if show_ocr_boxes:
            for box in ocr_boxes:
                x0, y0 = int(box.rect_img.x0), int(box.rect_img.y0)
                x1, y1 = int(box.rect_img.x1), int(box.rect_img.y1)
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (200, 120, 0), 1)

        # Draw redaction zones in red (semi-transparent)
        for r in rects_tl:
            x0, y0 = int(r.x0 * sx), int(r.y0 * sy)
            x1, y1 = int(r.x1 * sx), int(r.y1 * sy)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)

        pages.append({
            "page_index": i,
            "page_type": page_type.name,
            "redaction_count": len(rects_tl),
            "matches": matches,
            "annotated_bgr": annotated,
        })

    doc.close()
    return pages


def _iter_pdfs(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.rglob("*.pdf")):
        if p.name.startswith("."):
            continue
        yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch redact PDFs using OCR/layout rules.")
    parser.add_argument("--input-dir", required=True, help="Folder containing PDFs.")
    parser.add_argument("--output-dir", required=True, help="Where to write redacted PDFs.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional limit for testing.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cfg_path = Path(args.config).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input dir does not exist: {input_dir}")
    if not cfg_path.exists():
        raise SystemExit(f"Config does not exist: {cfg_path}")

    cfg = _load_config(cfg_path)

    # Initialize OCR once
    reader = easyocr.Reader(cfg.ocr_langs, gpu=True)

    pdfs = list(_iter_pdfs(input_dir))
    if args.max_files and args.max_files > 0:
        pdfs = pdfs[: args.max_files]

    if not pdfs:
        raise SystemExit(f"No PDFs found under: {input_dir}")

    for pdf in tqdm(pdfs, desc="Redacting PDFs"):
        out_name = pdf.stem + cfg.output_suffix + pdf.suffix
        out_path = output_dir / out_name
        redact_pdf(pdf, out_path, cfg, reader)

    print(f"Done. Wrote {len(pdfs)} redacted PDF(s) to: {output_dir}")


if __name__ == "__main__":
    main()

