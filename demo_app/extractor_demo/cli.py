from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from .config import AppConfig
from .providers import build_provider


def iter_pdfs(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.rglob("*.pdf") if not p.name.startswith(".")])


def ensure_all_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    out = {k: row.get(k, "") for k in fields}
    # Keep extras too (debugging), but don't collide with schema
    for k, v in row.items():
        if k not in out:
            out[k] = v
    return out


def maybe_redact(cfg: AppConfig, input_dir: Path) -> Path:
    if not cfg.pipeline.enable_redaction:
        return input_dir

    redacted_dir = input_dir.parent / f"{input_dir.name}.redacted"
    redacted_dir.mkdir(parents=True, exist_ok=True)

    cmd = cfg.pipeline.redaction_command.format(
        input_dir=str(input_dir),
        redacted_dir=str(redacted_dir),
        redaction_config=str(Path(cfg.pipeline.redaction_config).resolve()),
    )
    # Run through shell for the formatted command string.
    subprocess.check_call(cmd, shell=True)
    return redacted_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fields from PDFs into a table.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-file", required=True, help="out.xlsx or out.csv")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    cfg = AppConfig.model_validate(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    effective_input_dir = maybe_redact(cfg, input_dir)

    pdfs = iter_pdfs(effective_input_dir)
    if args.max_files and args.max_files > 0:
        pdfs = pdfs[: args.max_files]
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {effective_input_dir}")

    provider = build_provider(cfg)
    fields = cfg.schema.fields

    rows: list[dict[str, Any]] = []
    for pdf in tqdm(pdfs, desc="Extracting"):
        try:
            data = provider.extract(str(pdf), fields)
            data["_source_pdf"] = str(pdf)
            rows.append(ensure_all_fields(data, fields))
        except Exception as e:
            rows.append(ensure_all_fields({"_source_pdf": str(pdf), "needs_review": True, "_error": str(e)}, fields))

    df = pd.DataFrame(rows)
    out = Path(args.output_file).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    elif out.suffix.lower() in [".xlsx", ".xlsm"]:
        df.to_excel(out, index=False)
    else:
        raise SystemExit("Output file must be .csv or .xlsx")

    print(f"Wrote {len(df)} row(s) to {out}")


if __name__ == "__main__":
    main()

