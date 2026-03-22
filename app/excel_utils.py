from __future__ import annotations

"""
Excel read/write utilities for the app layer.

- parse_labels(): read 1-2026-DANE.xlsx → dict keyed by nr_wniosku
- save_result(): append an extraction result to EXTRACTED-DATA-TEMPLATE.xlsx
- compare_result(): field-by-field diff between VLM output and Excel ground truth
"""

import json
import re
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.styles import Font, PatternFill

BACKUP_DIR = Path("/Users/michalglomski/Documents/ARCH-APP-Project-Files-Backup")
EXCEL_LABELS = BACKUP_DIR / "1-2026-DANE.xlsx"


# ---------------------------------------------------------------------------
# Column indices (0-based)
# ---------------------------------------------------------------------------
COL_NR, COL_SPOSOB, COL_FLAGA = 0, 1, 2
COL_NAZWA, COL_ADRES, COL_TEREN, COL_POW_CAL = 3, 4, 5, 6
COL_SZEROK, COL_POW_NAD, COL_POW_POD = 7, 8, 9
COL_WYS_KRAW, COL_WYS_ZAB = 10, 11
COL_KOND_NAD, COL_KOND_POD, COL_DACH, COL_MEDIA = 12, 13, 14, 15


def _s(v) -> str:
    return "" if v is None else str(v).strip()


def _split(cell: str) -> tuple[str, str]:
    m = re.match(r'^([^:]+?):\s*(.*)$', cell)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", cell)


def _is_header(row) -> bool:
    return _s(row[COL_NR]).lower() == "nr wniosku"


def parse_labels() -> dict[str, dict]:
    """Parse the master Excel into {nr_wniosku: record}."""
    if not EXCEL_LABELS.exists():
        return {}
    wb = openpyxl.load_workbook(str(EXCEL_LABELS), read_only=True, data_only=True)
    ws = wb.active
    records: dict[str, dict] = {}
    current = None

    for row in ws.iter_rows(values_only=True):
        if all(v is None for v in row) or _is_header(row):
            continue
        nr_raw = row[COL_NR]
        if nr_raw is not None and _s(nr_raw) not in ("", "None"):
            try:
                nr = str(int(float(_s(nr_raw))))
            except ValueError:
                continue
            current = {
                "nr_wniosku": nr,
                "sposob_wypelnienia": _s(row[COL_SPOSOB]),
                "flaga_7_9": _s(row[COL_FLAGA]),
                "nazwa_inwestycji": _s(row[COL_NAZWA]),
                "adres": _s(row[COL_ADRES]),
                "teren_inwestycji": _s(row[COL_TEREN]),
                "pow_zabudowy_calosc": _s(row[COL_POW_CAL]),
                "budynki": [],
                "media": [],
            }
            records[nr] = current
            _add_building(current, row)
            m = _s(row[COL_MEDIA])
            if m:
                current["media"].append(m)
        elif current is not None:
            _add_building(current, row)
            m = _s(row[COL_MEDIA])
            if m:
                current["media"].append(m)

    wb.close()
    return records


def _add_building(record: dict, row) -> None:
    raw = _s(row[COL_SZEROK])
    if not raw:
        return
    label, val = _split(raw)

    def v(col):
        _, x = _split(_s(row[col]))
        return x

    record["budynki"].append({
        "oznaczenie": label or raw,
        "szerokosc_elewacji": val or raw,
        "suma_pow_nadziemnych": v(COL_POW_NAD),
        "suma_pow_podziemnych": v(COL_POW_POD),
        "wys_gornej_krawedzi": v(COL_WYS_KRAW),
        "wysokosc_zabudowy": v(COL_WYS_ZAB),
        "ilosc_kond_nadziemnych": v(COL_KOND_NAD),
        "ilosc_kond_podziemnych": v(COL_KOND_POD),
        "geometria_dachu": v(COL_DACH),
    })


# ---------------------------------------------------------------------------
# Save result to EXTRACTED-DATA-TEMPLATE.xlsx
# ---------------------------------------------------------------------------

TEMPLATE_PATH = Path(__file__).parent.parent / "Project Files" / "EXTRACTED-DATA-TEMPLATE.xlsx"
OUTPUT_PATH   = Path(__file__).parent / "extracted_results.xlsx"


def save_result(pred: dict) -> str:
    """Append a prediction to the output Excel file. Returns the output path."""
    # Load or create workbook
    if OUTPUT_PATH.exists():
        wb = openpyxl.load_workbook(str(OUTPUT_PATH))
        ws = wb.active
    else:
        wb = openpyxl.load_workbook(str(TEMPLATE_PATH))
        ws = wb.active

    nr      = pred.get("nr_wniosku", "")
    sposob  = pred.get("sposob_wypelnienia", "")
    flaga   = pred.get("flaga_7_9", "")
    nazwa   = pred.get("nazwa_inwestycji", "")
    adres   = pred.get("adres", "")
    teren   = pred.get("teren_inwestycji", "")
    pow_cal = pred.get("pow_zabudowy_calosc", "")
    budynki = pred.get("budynki") or []
    media   = pred.get("media") or []

    n_rows = max(len(budynki), len(media), 1)
    for i in range(n_rows):
        b = budynki[i] if i < len(budynki) else {}
        m = media[i]   if i < len(media)   else ""
        oz = b.get("oznaczenie", "") if b else ""
        ws.append([
            nr     if i == 0 else "",
            sposob if i == 0 else "",
            flaga  if i == 0 else "",
            nazwa  if i == 0 else "",
            adres  if i == 0 else "",
            teren  if i == 0 else "",
            pow_cal if i == 0 else "",
            f"{oz}: {b.get('szerokosc_elewacji','')}"    if b else "",
            f"{oz}: {b.get('suma_pow_nadziemnych','')}"  if b else "",
            f"{oz}: {b.get('suma_pow_podziemnych','')}"  if b else "",
            f"{oz}: {b.get('wys_gornej_krawedzi','')}"   if b else "",
            f"{oz}: {b.get('wysokosc_zabudowy','')}"     if b else "",
            f"{oz}: {b.get('ilosc_kond_nadziemnych','')}" if b else "",
            f"{oz}: {b.get('ilosc_kond_podziemnych','')}" if b else "",
            f"{oz}: {b.get('geometria_dachu','')}"        if b else "",
            m,
        ])

    wb.save(str(OUTPUT_PATH))
    return str(OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Compare VLM prediction against Excel ground truth
# ---------------------------------------------------------------------------

FLAT_FIELDS = [
    "nr_wniosku", "sposob_wypelnienia", "flaga_7_9",
    "nazwa_inwestycji", "adres", "teren_inwestycji", "pow_zabudowy_calosc",
]
BUILDING_FIELDS = [
    "szerokosc_elewacji", "suma_pow_nadziemnych", "suma_pow_podziemnych",
    "wys_gornej_krawedzi", "wysokosc_zabudowy",
    "ilosc_kond_nadziemnych", "ilosc_kond_podziemnych", "geometria_dachu",
]


def _norm(v: Any) -> str:
    return "" if v is None else str(v).strip().lower()


def compare_result(pred: dict, gold: dict) -> dict:
    """
    Returns a comparison dict:
    {
      "flat": { "adres": {"pred": "...", "gold": "...", "match": true}, ... },
      "budynki": [ { "szerokosc_elewacji": {...}, ... }, ... ],
      "media": { "pred": [...], "gold": [...], "match": true },
      "overall_accuracy": 0.85
    }
    """
    result: dict[str, Any] = {"flat": {}, "budynki": [], "media": {}}
    correct = total = 0

    for f in FLAT_FIELDS:
        p, g = _norm(pred.get(f)), _norm(gold.get(f))
        match = p == g
        result["flat"][f] = {"pred": pred.get(f, ""), "gold": gold.get(f, ""), "match": match}
        correct += int(match); total += 1

    pred_blds = pred.get("budynki") or []
    gold_blds = gold.get("budynki") or []
    for i in range(max(len(pred_blds), len(gold_blds))):
        pb = pred_blds[i] if i < len(pred_blds) else {}
        gb = gold_blds[i] if i < len(gold_blds) else {}
        bld_cmp = {}
        for f in BUILDING_FIELDS:
            p, g = _norm(pb.get(f)), _norm(gb.get(f))
            match = p == g
            bld_cmp[f] = {"pred": pb.get(f, ""), "gold": gb.get(f, ""), "match": match}
            correct += int(match); total += 1
        result["budynki"].append(bld_cmp)

    pred_media = set(_norm(m) for m in (pred.get("media") or []))
    gold_media = set(_norm(m) for m in (gold.get("media") or []))
    match = pred_media == gold_media
    result["media"] = {
        "pred": pred.get("media") or [],
        "gold": gold.get("media") or [],
        "match": match,
    }
    correct += int(match); total += 1

    result["overall_accuracy"] = round(correct / max(1, total), 3)
    return result
