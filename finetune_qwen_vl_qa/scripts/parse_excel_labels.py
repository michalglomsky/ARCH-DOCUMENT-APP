from __future__ import annotations

"""
Parse the master Excel spreadsheet (e.g. 1-2026-DANE.xlsx) into a JSON file
mapping nr_wniosku → structured extraction record.

The Excel has a multi-row layout per permit:
  - The first row of each permit contains the permit number and all flat fields.
  - Continuation rows (nr_wniosku column is empty) add extra buildings and media entries.

Output: labels.json  — a dict keyed by str(nr_wniosku):
  {
    "808": {
      "nr_wniosku": "808",
      "sposob_wypelnienia": "Komputerowo",
      "flaga_7_9": "BRAK",
      "nazwa_inwestycji": "BUDOWA ...",
      "adres": "Łąki 97, 08-404 Górzno",
      "teren_inwestycji": "teren części działki ...",
      "pow_zabudowy_calosc": "od 90,0 m2 do 350,0 m2",
      "budynki": [
        {
          "oznaczenie": "1. Mieszkalny",
          "szerokosc_elewacji": "od 8,0 do 25,0 m",
          "suma_pow_nadziemnych": "od 90,0 do 280,0 m2",
          "suma_pow_podziemnych": "nie dotyczy",
          "wys_gornej_krawedzi": "od 2,5 do 9,0 m",
          "wysokosc_zabudowy": "od 5,0 do 9,0 m",
          "ilosc_kond_nadziemnych": "max 2",
          "ilosc_kond_podziemnych": "1",
          "geometria_dachu": "1/2/4/wielo (15-45°)"
        }
      ],
      "media": ["Woda: studnia", "Prąd: z sieci", "Ciepło: indyw. kotłownia", "Ścieki: indyw. oczyszczalnia"]
    },
    ...
  }

Usage:
    source finetune_qwen_vl_pytorch/.venv311/bin/activate

    python finetune_qwen_vl_qa/scripts/parse_excel_labels.py \
        --excel "Project Files/1-2026-DANE.xlsx" \
        --output finetune_qwen_vl_qa/data/labels.json
"""

import argparse
import json
import re
from pathlib import Path

import openpyxl


# ---------------------------------------------------------------------------
# Column indices (0-based) — matches the Excel layout in 1-2026-DANE.xlsx
# ---------------------------------------------------------------------------
COL_NR          = 0   # Nr wniosku
COL_SPOSOB      = 1   # Sposób wypełnienia
COL_FLAGA       = 2   # Flaga 7.9
COL_NAZWA       = 3   # Nazwa inwestycji
COL_ADRES       = 4   # Adres
COL_TEREN       = 5   # Teren inwestycji
COL_POW_CAL     = 6   # Pow. zabudowy (całość)
COL_SZEROK      = 7   # Szerokość elewacji     ← per-building start
COL_POW_NAD     = 8   # Suma pow. nadziemnych
COL_POW_POD     = 9   # Suma pow. podziemnych
COL_WYS_KRAW    = 10  # Wys. górnej krawędzi
COL_WYS_ZAB     = 11  # Wysokość zabudowy
COL_KOND_NAD    = 12  # Ilość kond. nadziemnych
COL_KOND_POD    = 13  # Ilość kond. podziemnych
COL_DACH        = 14  # Geometria dachu        ← per-building end
COL_MEDIA       = 15  # Media


def _str(v) -> str:
    """Convert any cell value to a stripped string, empty string if None."""
    if v is None:
        return ""
    return str(v).strip()


def _split_label_value(cell_text: str) -> tuple[str, str]:
    """
    Split "1. Mieszkalny: od 8,0 do 25,0 m" → ("1. Mieszkalny", "od 8,0 do 25,0 m").
    If no colon found, returns ("", cell_text).
    Handles "[Nie dotyczy - ...]" style cells without a colon.
    """
    # Match "N. Label: value"  or  "N. Abbrev: value"
    m = re.match(r'^([^:]+?):\s*(.*)$', cell_text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", cell_text


def _parse_building_row(row_vals: tuple) -> dict | None:
    """
    Parse the per-building columns from a single Excel row.
    Returns None if the building section is entirely empty / not applicable.
    """
    szerok_raw = _str(row_vals[COL_SZEROK])
    if not szerok_raw:
        return None  # no building info on this row

    label, szerok_val = _split_label_value(szerok_raw)

    def _val(col: int) -> str:
        raw = _str(row_vals[col])
        _, v = _split_label_value(raw)
        return v if v else raw

    return {
        "oznaczenie":            label or szerok_raw,
        "szerokosc_elewacji":    szerok_val if szerok_val else szerok_raw,
        "suma_pow_nadziemnych":  _val(COL_POW_NAD),
        "suma_pow_podziemnych":  _val(COL_POW_POD),
        "wys_gornej_krawedzi":   _val(COL_WYS_KRAW),
        "wysokosc_zabudowy":     _val(COL_WYS_ZAB),
        "ilosc_kond_nadziemnych": _val(COL_KOND_NAD),
        "ilosc_kond_podziemnych": _val(COL_KOND_POD),
        "geometria_dachu":       _val(COL_DACH),
    }


def _is_header_row(row_vals: tuple) -> bool:
    """True for the repeating header rows embedded in the sheet."""
    v = _str(row_vals[COL_NR])
    return v.lower() in ("nr wniosku", "nr wniosku")


def parse_excel(xlsx_path: Path) -> dict[str, dict]:
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb.active

    records: dict[str, dict] = {}
    current: dict | None = None

    for row in ws.iter_rows(values_only=True):
        if all(v is None for v in row):
            continue
        if _is_header_row(row):
            continue

        nr_raw = row[COL_NR]

        # --- New permit record starts when col 0 has a numeric value ---
        if nr_raw is not None and str(nr_raw).strip() not in ("", "None"):
            try:
                nr_int = int(float(str(nr_raw).strip()))
            except ValueError:
                continue  # non-numeric in col 0 → skip (merged header etc.)

            nr_str = str(nr_int)
            current = {
                "nr_wniosku":         nr_str,
                "sposob_wypelnienia": _str(row[COL_SPOSOB]),
                "flaga_7_9":          _str(row[COL_FLAGA]),
                "nazwa_inwestycji":   _str(row[COL_NAZWA]),
                "adres":              _str(row[COL_ADRES]),
                "teren_inwestycji":   _str(row[COL_TEREN]),
                "pow_zabudowy_calosc": _str(row[COL_POW_CAL]),
                "budynki":            [],
                "media":              [],
            }
            records[nr_str] = current

            bld = _parse_building_row(row)
            if bld:
                current["budynki"].append(bld)

            media = _str(row[COL_MEDIA])
            if media:
                current["media"].append(media)

        # --- Continuation row: belongs to the last seen permit ---
        elif current is not None:
            bld = _parse_building_row(row)
            if bld:
                current["budynki"].append(bld)
            media = _str(row[COL_MEDIA])
            if media:
                current["media"].append(media)

    wb.close()
    return records


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse Excel labels into a JSON mapping nr_wniosku → record."
    )
    ap.add_argument("--excel", required=True, help="Path to the Excel label file (e.g. 1-2026-DANE.xlsx)")
    ap.add_argument("--output", required=True, help="Output path for labels.json")
    args = ap.parse_args()

    xlsx = Path(args.excel).expanduser().resolve()
    out  = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    records = parse_excel(xlsx)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Parsed {len(records)} permit records → {out}")
    # Quick stats
    n_with_buildings = sum(1 for r in records.values() if r["budynki"])
    n_multi_bld      = sum(1 for r in records.values() if len(r["budynki"]) > 1)
    print(f"  with building data:    {n_with_buildings}")
    print(f"  multi-building forms:  {n_multi_bld}")


if __name__ == "__main__":
    main()
