## PDF Redaction Tool (Layout/OCR-based)

Batch-redacts sensitive regions in PDFs **before** they are shared with an LLM.

This tool uses **OCR + layout cues** (Approach B) to find page types and anchor text, then applies **true PDF redactions** (not just visual overlays) using PyMuPDF.

### What it does

- **Batch process** a folder of PDFs
- **Detect page type** by searching OCR text for keywords/regex patterns (works even if pages are reordered)
- **Find anchors** (labels like “Wnioskodawca”, “Imię i nazwisko”, “PESEL”, etc.) via OCR bounding boxes
- **Redact configurable regions** relative to those anchors (plus optional fixed boxes)
- Output redacted PDFs to an output folder

### Why this approach

- Works with **scans/OCR** where native PDF text selection is missing
- Handles **small shifts** because redaction regions are computed relative to detected text boxes
- Handles **page reordering** by classifying pages based on their content

---

## Install

Create a virtualenv (recommended) and install dependencies:

```bash
cd redaction_tool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `easyocr` pulls in PyTorch. This is the simplest OCR path but can be heavy.
- If you want a lighter install later, we can swap OCR to Tesseract + `pytesseract` (requires `brew install tesseract`).

---

## Configure redactions

Edit `config.example.yaml` and copy it to `config.yaml`:

```bash
cp config.example.yaml config.yaml
```

You control:

- OCR language(s)
- DPI used for rendering pages before OCR
- How page types are recognized (keywords/regex)
- What regions are redacted (relative to anchor text boxes, or fixed coordinates)

---

## Run on your example PDFs

Your example PDFs are in `Project Files/` (e.g. `945 wniosek.pdf`).

From `OpenClawConfig/`:

```bash
source redaction_tool/.venv/bin/activate
python -m redaction_tool.redactor \
  --input-dir "Project Files" \
  --output-dir "Project Files/redacted" \
  --config redaction_tool/config.yaml
```

---

## How to define “what to redact” (adjustable)

Redactions are defined as rules per **page type**. Each rule can:

- Find an anchor text box by regex (OCR result)
- Apply one or more rectangles:
  - **Relative** to the anchor box (`dx, dy, w, h`)
  - **Absolute** in PDF-page coordinates (`x0, y0, x1, y1`) as a fallback

This is designed so you can change “what to cover” by editing YAML only.

---

## Output safety note

This tool applies PDF redactions using PyMuPDF’s redaction annotations and then **applies** them (content removed). This is safer than simply drawing black rectangles.

---

## Next steps / tuning

After one run, open a few redacted PDFs to verify coverage.
Typical tuning loop:

- Add/adjust anchor regexes (Polish labels)
- Add/adjust offsets for your form template
- Add additional page types (Załącznik A/B/C, etc.)

