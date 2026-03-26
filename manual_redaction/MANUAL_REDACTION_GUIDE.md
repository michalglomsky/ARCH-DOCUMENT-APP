# Manual Redaction Tool

A browser-based UI for drawing, managing, and applying PDF redaction zones manually.
Runs as a local FastAPI server on port 8083.

---

## Start the server

```bash
./run_manual_redaction.sh
# or directly:
/path/to/.venv/bin/python3 manual_redaction/app.py --port 8083
```

Then open `http://127.0.0.1:8083` in your browser.

---

## Features

### Loading documents

- Type a path (file or directory) in the **PDF file or directory** field and click **Load**.
- Or click **📂** to open the file explorer (supports multi-select — see below).
- Single PDF or entire folder — all PDFs appear in the left file list.

### File explorer (multi-select)

When browsing to load PDFs:

| Action | Effect |
|--------|--------|
| Click | Select (clears previous selection) |
| Ctrl+click | Toggle file in/out of selection |
| Shift+click | Select a contiguous range |
| Double-click | Load that single file immediately |
| ✓ Load selected (N) | Load all N highlighted files |
| Select this folder | Load every PDF in the current directory |
| 📁 New folder | Create a new subdirectory (useful for output) |

### Drawing redaction boxes

1. Switch to **✏️ Draw** mode (keyboard: `D`).
2. Click and drag on the document to draw a box.
3. Boxes are displayed in red; they are saved as solid black in the exported PDF.

### Moving & resizing

Switch to **✋ Move** mode (`M`). Click a box to select it — 8 handles appear:

- Drag a **corner handle** to resize diagonally.
- Drag an **edge handle** to resize in one axis.
- Drag the **box body** to reposition.

### Erasing

Switch to **🧹 Erase** mode (`E`). Click any box to delete it instantly. Supports undo.

### Undo / Redo

- **↩ Undo** (`Ctrl+Z`) — up to 60 steps.
- **↪ Redo** (`Ctrl+Y`).

### Zoom

- **－ / ＋** buttons or keyboard `-` / `+` — steps of 25%.
- Presets: 25%, 50%, 75%, 100%.
- **Fit** — scales the page to fill the editor width.
- Default zoom is **50%**.

### Template system

Click **Template ▾** to open the menu:

| Item | Action |
|------|--------|
| 💾 Save as template | Captures all zones from the current document |
| 📋 Apply to this document | Stamps template onto current doc (with confirmation) |
| 📋 Apply to all documents | Stamps template onto every file in the list |
| ✕ Clear template | Discards the saved template |
| 🎯 Set intelligent alignment | Draw an anchor region + type anchor text (see below) |
| ✕ Clear alignment | Removes the alignment anchor |

The badge on the button shows how many zones are in the current template.

### Intelligent alignment anchor

Allows the template to shift automatically to match each document's layout.

1. Open **Template ▾ → 🎯 Set intelligent alignment**.
2. The canvas enters blue crosshair mode — draw a box around a text label that appears consistently across all your documents (e.g. "Wniosek").
3. A popup appears — type the anchor text and click **✓ Set anchor**.
4. The anchor region is shown as a blue dashed ⚓ box on the canvas.

When applying the template, the backend searches for the anchor text **only inside the drawn region** (faster and more precise than full-page search). It finds the text in both the template document and each target document, then shifts all zones linearly by the difference.

- Text-based PDFs: instant search via PyMuPDF's built-in index.
- Scanned/image PDFs: automatic EasyOCR fallback.
- Comma-separated candidates are tried in order: `wniosek, Wnioskodawca`.
- If the anchor cannot be found in a target, zones are stamped at original positions.

### Zones panel

The right panel lists every box on the current page. Click a row to select/highlight the zone on the canvas. Click **✕** to delete a specific zone.

### Saving redacted files

- **✂️ Save** — applies zones to the current document and writes to the output directory.
- **✂️ Redact & Save All** — processes every document that has at least one zone.

Original files are never modified. Output always goes to the **Output directory** field.
The output directory browser also supports **📁 New folder**.

### File list management

- The active document is highlighted in orange.
- Each file shows its zone count, or `done`/`error` after saving.
- Click **✕** on a file row to remove it from the session (file on disk is untouched).

### Session persistence

All zones, file paths, the template, and the alignment anchor are saved to `localStorage` after every change. Refreshing or closing the tab preserves your work.

- **🗑 New session** — clears everything and starts fresh (asks for confirmation).

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `D` | Draw mode |
| `M` | Move mode |
| `E` | Erase mode |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `+` | Zoom in |
| `-` | Zoom out |
| `Esc` | Close any open modal |

---

## API (backend)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve the UI |
| POST | `/render` | Render PDF pages to base64 JPEG |
| POST | `/list_pdfs` | List PDF filenames in a directory (top-level only) |
| POST | `/browse` | List directory contents for the file explorer |
| POST | `/mkdir` | Create a directory |
| POST | `/redact_save` | Apply zones to a single PDF and save |
| POST | `/batch_redact` | Apply zones to multiple PDFs in one call |
| POST | `/align_zones` | Shift template zones using anchor text alignment |
| GET | `/health` | Health check |

All zone coordinates are **normalised (0–1)** so they transpose correctly across documents with different page sizes.

---

## Implementation notes

- **`images=0`** in `apply_redactions` — draws black fill without removing underlying image data, preventing large black areas in scanned PDFs.
- The anchor region search (`anchor_region` in `AlignRequest`) clips PyMuPDF's `page.search_for()` to a small rectangle, making it much faster than full-page search.
- EasyOCR is lazy-loaded on first use and reused across requests (`_ocr_reader` global).
