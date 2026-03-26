# Fine-Tuning Guide — Qwen2.5-VL on Construction Permit Extraction

This document captures the prompt engineering decisions, annotation strategy, and
enhancement roadmap for fine-tuning the VLM on Polish construction permit forms
(wnioski o warunki zabudowy / pozwolenie na budowę).

---

## How the VLM is currently prompted

The extraction prompt lives in `scripts/qa_utils.py` (`_QA_PROMPT`).  It is built
once at server startup and reused for every request — no part of it comes from the
app or the user.

At inference time the model receives a single user message structured as:

```
[image: page 1]  [image: page 2]  ...  [image: page N]  [text: extraction prompt]
```

All pages are embedded as visual tokens in **one** message.  The model sees the
entire document at once and outputs a single JSON object.

The prompt does three things:
1. Sets the role and task context (Polish permit forms)
2. Defines the exact JSON schema the model must follow
3. Lists rules for edge cases (illegible text, missing fields, infrastructure forms)

**This prompt is the contract between you and the model.**  Whatever schema and
special tokens (`[BRAK]`, `[NIECZYTELNE]`, `[Nie dotyczy]`) are defined here must
be reproduced consistently in every training label.  If a label uses `"-"` where
the prompt says `"[BRAK]"`, the model receives contradictory training signal.

---

## Stage 1 — Before generating any training data

### 1.1  Lock the prompt

Do not change the prompt text after you start generating QA pairs.  Even a minor
wording change invalidates previously generated pairs because the model was trained
to respond to a specific input.

### 1.2  Clean the Excel ground truth

Run the app's Compare function on a random sample of ~20 documents before building
the dataset.  Look for:

- Fields stored as `"-"`, `"N/A"`, `"brak"` instead of `"[BRAK]"` — normalise them
- Numeric values stored inconsistently (`"120 m2"` vs `"120,0 m2"` vs `120`) — pick one format and apply it everywhere
- Missing `oznaczenie` on building rows — the model needs a label to anchor each building
- Extra whitespace or line breaks in cell values — strip them

Label quality beats label quantity.  20 perfectly clean labels outperform 97 noisy ones.

### 1.3  Decide on empty-field representation

Pick one convention and apply it uniformly across all 121 Excel rows:

| Situation | Use |
|-----------|-----|
| Field not present on this form | `[BRAK]` |
| Field present but illegible | `[NIECZYTELNE]` |
| Form is infrastructure (no building) | `[Nie dotyczy]` in building fields |
| Field present and readable | exact text as printed |

---

## Stage 2 — Building the dataset (97 / 12 / 12 split)

### 2.1  Dataset split

| Split | Count | Purpose |
|-------|-------|---------|
| Train | 97 | Gradient updates |
| Validation | 12 | Loss monitoring / early stopping |
| Test (held-out) | 12 | Final accuracy benchmark — never seen during training |

### 2.2  Stratify the splits

Do not split randomly.  Make sure each split contains a representative mix of:

- Single-building vs. multi-building documents
- Komputerowo (typed) vs. Ręcznie (handwritten) forms
- Clean scans vs. poor-quality / stamped scans
- Documents where `needs_review` is true

If your training set contains only clean, easy documents, the model will fail on the
hard ones that actually matter.

### 2.3  Each QA pair format

```json
{
  "pdf": "wz_808.pdf",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "file:///abs/path/wz_808_p001.png"},
        {"type": "image", "image": "file:///abs/path/wz_808_p002.png"},
        {"type": "text",  "text": "<the full extraction prompt>"}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"nr_wniosku\": \"808\", \"adres\": \"...\", ...}"
    }
  ]
}
```

The assistant turn is the cleaned Excel label serialised as JSON — this is what the
model is trained to reproduce.

---

## Stage 3 — Prompt engineering enhancements

### 3.1  Cross-page verification rule (already added)

Polish permit forms are not standardised across municipalities.  The same field
(`nr_wniosku`, `adres`, building parameters) can appear in the form header, the
body table, an attached stamp, a cover letter, or a handwritten note — on any page.

The current prompt now includes:

> Fields often appear in multiple locations across pages (headers, tables, stamps,
> cover letters, attachments).  Scan ALL pages for each field and use the most
> legible occurrence to determine the correct value.
>
> If the same field has conflicting values in different locations, prefer printed or
> typed text over handwriting, and set needs_review to true.

**Why this works better than page-number hints:**
Hardcoding "nr_wniosku is on page 1" would teach the model a rule that breaks the
moment it encounters a non-standard form layout.  The cross-verification rule instead
teaches the model a behaviour — scan everything, prefer clarity — which generalises
across all form variants.

### 3.2  What NOT to add to the static prompt

| Idea | Why to skip it |
|------|---------------|
| "Field X is typically on page Y" | Forms are not standardised; false hints hurt more than they help |
| Long explanations of Polish bureaucratic context | Increases prompt token count with marginal benefit |
| Example JSON in the prompt | The schema definition already covers this; examples belong in training data |

---

## Stage 4 — Iteration 2 enhancements (after first fine-tuning run)

Only add these after evaluating your first fine-tuned model on the held-out test set
and identifying specific failure modes.

### 4.1  Per-field source page annotation

If accuracy is still low on multi-page documents, annotate which pages each field
was read from in the training labels:

```json
{
  "nr_wniosku":  {"value": "808",                "source_pages": [1, 3]},
  "adres":       {"value": "ul. Przykładowa 5",  "source_pages": [1]},
  "budynki":     [{"oznaczenie": "1. Mieszkalny", "source_pages": [2, 3]}]
}
```

The model learns the statistical layout of your specific corpus rather than relying
on general assumptions.  This is significantly more annotation work — only do it if
the first run reveals page-level confusion as a clear failure pattern.

### 4.2  Increase image resolution

Current render DPI is 200.  For documents with small numeric values (floor areas,
heights, facade widths), increasing to 250–300 DPI during both training and inference
measurably improves reading accuracy.  This requires re-rendering all training images
and re-running inference with the same setting.

Change in `qa_utils.py`:
```python
def render_pdf_to_tmp(pdf_path, dpi=250, max_pages=8):  # was 200
```

Apply the same change in the training dataset generation script so training and
inference use identical image quality.

### 4.3  Targeted hard examples

After evaluating on the test set, identify the field types with the lowest accuracy.
Add 5–10 additional training examples that specifically contain those difficult cases:

- Multi-building forms with 3+ buildings
- Heavily stamped or annotated pages
- Forms with handwritten corrections over typed text
- Infrastructure-only forms (`[Obiekt inżynierski]`)

Adding targeted hard examples is usually more effective than adding more random
examples once you are past ~50 training samples.

### 4.4  LoRA hyperparameter tuning

Starting recommended configuration:

```
r           = 16
lora_alpha  = 32
target      = q_proj, k_proj, v_proj, o_proj
epochs      = 3
early stop  = validation loss not improving for 2 consecutive epochs
```

If the model overfits (train loss falling, val loss rising), reduce `r` to 8 or
add dropout (`lora_dropout=0.05`).  If underfitting (both losses plateau early),
increase `r` to 32 or train for more epochs.

---

## Stage 5 — Evaluating results

Use the app's Compare function (⚖ Compare button) against the held-out 12 test
documents to get a field-by-field accuracy breakdown.  The `score_prediction()`
function in `qa_utils.py` computes the same metric programmatically for batch
evaluation.

**Common failure patterns and their causes:**

| Symptom | Likely cause |
|---------|-------------|
| JSON parse errors | Model not following output format — check prompt hasn't changed |
| Wrong values but correct structure | Label noise in training data — review Excel ground truth |
| Good on simple forms, bad on multi-building | Need harder training examples |
| `needs_review` always true | Model is uncertain — usually a sign of insufficient training data or too-low DPI |
| Flat fields correct, building fields wrong | Building table parsing is hard — consider increasing max_pages or DPI |

---

## Adaptive preprocessing

Based on: *Input-Adaptive Visual Preprocessing for Efficient Fast Vision-Language Model Inference*
(Cahyani et al., University of Brawijaya).

### What it does

Two techniques are applied per page before the image is passed to the VLM:

1. **Blank-page detection** — computes a complexity score [0–1] from edge density and
   Shannon entropy.  Pages below `blank_threshold` are skipped entirely.
   Disabled by default (`BLANK_THRESHOLD = 0.0`) because permit form PDFs are assumed
   to contain only relevant pages.  Raise to 0.15 if you ever encounter PDFs with
   blank separators or cover sheets.

2. **Margin cropping** — finds the bounding box of non-white content pixels and crops
   to it (with 2 % padding).  Construction permit forms typically have 10–20 % white
   margin on each side; removing it reduces the spatial extent of the image and therefore
   the number of visual tokens the model must process.

Expected gain for construction permits: **15–35 % fewer visual tokens** and proportionally
faster inference.  Lower than the paper's 55–60 % because permit pages are uniformly
high-complexity (dense text, tables, stamps) so few pages are skipped.

### Consistency requirement

**The same preprocessing must be used for both training and inference.**
If you enable `--preprocess` in `build_dataset.py`, you must also start the VLM server
with `--preprocess`.  Mismatching the two causes a distribution shift between training
images and inference images, which will hurt accuracy.

### Inspecting the results

#### From the dataset builder

Run with `--preprocess` and compare the two image folders side-by-side:

```
finetune_qwen_vl_qa/data/
  images/                      ← original renders (always written)
    wz_808_p001.png
    wz_808_p002.png
    ...
  images_preprocessed/         ← produced only with --preprocess
    wz_808/
      wz_808_p001.png          ← cropped version (open to compare with original)
      wz_808_p002.png
      preprocess_stats.json    ← per-page decisions for this document
    wz_1024/
      ...
```

Open `preprocess_stats.json` for any document to see exactly what the preprocessor decided:

```json
[
  {
    "file": "wz_808_p001.png",
    "complexity": 0.412,
    "original_size": [1240, 1754],
    "decision": "cropped",
    "crop_box": [42, 38, 1198, 1716],
    "processed_size": [1156, 1678]
  },
  {
    "file": "wz_808_p002.png",
    "complexity": 0.089,
    "original_size": [1240, 1754],
    "decision": "skipped",
    "reason": "complexity 0.089 < threshold 0.15"
  }
]
```

Blank-page skipping is disabled by default (`BLANK_THRESHOLD = 0.0`).  If your corpus
ever includes PDFs with blank separators or cover sheets, raise this to 0.15.

#### From the inference server (app)

Start the VLM server with both flags:

```bash
python finetune_qwen_vl_qa/scripts/serve_vlm_qa.py \
    --preprocess \
    --preprocess-debug-dir finetune_qwen_vl_qa/data/preprocess_debug
```

After extracting a document through the app, the debug folder will contain:

```
finetune_qwen_vl_qa/data/preprocess_debug/
  wz_808/
    wz_808_p001.png          ← what the VLM actually saw
    wz_808_p002.png
    preprocess_stats.json    ← complexity scores and crop decisions
```

This lets you verify that the VLM received sensible images for every document
you extract through the app UI, without changing what the user sees in the PDF
viewer (which always renders the original PDF via PDF.js).

---

## Running the dataset generation script

```bash
# Without preprocessing (baseline)
python finetune_qwen_vl_qa/scripts/build_dataset.py \
    --excel   /Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup/1-2026-DANE.xlsx \
    --pdf-dir /Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup \
    --out-dir finetune_qwen_vl_qa/data

# With preprocessing (recommended once you have verified the results)
python finetune_qwen_vl_qa/scripts/build_dataset.py \
    --excel   /Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup/1-2026-DANE.xlsx \
    --pdf-dir /Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup \
    --out-dir finetune_qwen_vl_qa/data \
    --preprocess
```

Run with `--dry-run` first to check matching and validation without rendering anything:

```bash
python finetune_qwen_vl_qa/scripts/build_dataset.py \
    --excel   /path/to/labels.xlsx \
    --pdf-dir /path/to/pdfs \
    --out-dir finetune_qwen_vl_qa/data \
    --dry-run
```

What the script does:
1. Parses the Excel file using the same `parse_labels()` used by the app
2. Matches each PDF to its label by extracting the permit number from the filename (e.g. `wz_808.pdf` → `808`)
3. Validates every assistant JSON string before writing — warns on missing keys or empty `budynki`
4. Assigns each document a stratum (single/multi building × typed/handwritten) for balanced splits
5. Renders pages to `data/images/` at the configured DPI (default 200 — must match inference)
6. Writes `train.jsonl`, `val.jsonl`, `test.jsonl` and a `dataset_stats.txt` summary

Outputs:
```
finetune_qwen_vl_qa/data/
  images/              PNG pages for all matched documents
  train.jsonl          97 records
  val.jsonl            12 records
  test.jsonl           12 records
  skipped.txt          PDFs that had no matching Excel label
  dataset_stats.txt    Full summary with stratum breakdown and any warnings
```

Review `dataset_stats.txt` and `skipped.txt` before starting training.
Any validation warnings there indicate label issues that should be fixed in the Excel first.

---

## File reference

| File | Role |
|------|------|
| `scripts/build_dataset.py` | Dataset generation — renders PDFs, builds JSONL splits |
| `scripts/qa_utils.py` | Prompt definition, PDF rendering, inference utilities |
| `scripts/serve_vlm_qa.py` | FastAPI server that loads the model and serves `/extract`, `/query` |
| `app/excel_utils.py` | Excel label parsing and ground-truth comparison |
| `app/server.py` | App server — proxies extract/compare requests, serves the UI |
