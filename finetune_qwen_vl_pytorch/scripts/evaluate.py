from __future__ import annotations

"""
Batch evaluation of Qwen2.5-VL extraction on a labeled JSONL set.

Metrics reported:
  - Overall field accuracy (exact-match, case-insensitive)
  - Per-field accuracy
  - needs_review precision and recall

Usage:
    source finetune_qwen_vl_pytorch/.venv311/bin/activate

    # Baseline (zero-shot, no adapter):
    python finetune_qwen_vl_pytorch/scripts/evaluate.py \
        --val-jsonl finetune_qwen_vl_pytorch/data/val.jsonl \
        --model     Qwen/Qwen2.5-VL-7B-Instruct \
        --output    eval_baseline.json

    # After fine-tuning:
    python finetune_qwen_vl_pytorch/scripts/evaluate.py \
        --val-jsonl    finetune_qwen_vl_pytorch/data/val.jsonl \
        --model        Qwen/Qwen2.5-VL-7B-Instruct \
        --lora-adapter finetune_qwen_vl_pytorch/out/lora_run1 \
        --output       eval_finetuned.json

Records without a non-empty "target_json" are skipped (they're unlabeled).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from vlm_utils import build_prompt, parse_json_response, run_inference


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

SPECIAL_VALUES = {"[nieczytelne]", "[brak]", "[do weryfikacji]"}


def field_match(pred: Any, gold: Any) -> bool:
    """
    Exact match after stripping and lower-casing.
    Both special marker values (NIECZYTELNE etc.) are compared as-is.
    """
    p = str(pred).strip().lower()
    g = str(gold).strip().lower()
    return p == g


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: Any,
    processor: Any,
    samples: list[dict],
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    per_field_correct: dict[str, int] = {}
    per_field_total: dict[str, int] = {}
    sample_records: list[dict] = []

    nr_tp = nr_fp = nr_fn = 0  # needs_review stats

    for sample in tqdm(samples, desc="Evaluating"):
        gold = sample.get("target_json", {})
        if not gold:
            continue
        if isinstance(gold, str):
            try:
                gold = json.loads(gold)
            except json.JSONDecodeError:
                continue

        schema_fields = list(gold.keys())
        prompt = sample.get("prompt") or build_prompt(schema_fields)

        raw = run_inference(
            model,
            processor,
            image_path=sample["image_path"],
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        pred = parse_json_response(raw)

        if device == "mps":
            torch.mps.empty_cache()

        field_scores: dict[str, bool] = {}
        for field, gold_val in gold.items():
            if field == "needs_review":
                continue
            per_field_total[field] = per_field_total.get(field, 0) + 1
            correct = field_match(pred.get(field, ""), gold_val)
            per_field_correct[field] = per_field_correct.get(field, 0) + int(correct)
            field_scores[field] = correct

        # needs_review
        gold_nr = bool(gold.get("needs_review", False))
        pred_nr = bool(pred.get("needs_review", False))
        if gold_nr and pred_nr:
            nr_tp += 1
        elif not gold_nr and pred_nr:
            nr_fp += 1
        elif gold_nr and not pred_nr:
            nr_fn += 1

        sample_records.append({
            "image_path": sample["image_path"],
            "gold": gold,
            "pred": pred,
            "field_scores": field_scores,
            "needs_review_gold": gold_nr,
            "needs_review_pred": pred_nr,
        })

    # Aggregate
    all_fields = sorted(per_field_total.keys())
    field_accuracy = {
        f: per_field_correct.get(f, 0) / per_field_total[f]
        for f in all_fields
    }
    total_tokens = sum(per_field_total.values())
    total_correct = sum(per_field_correct.values())
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    nr_precision = nr_tp / (nr_tp + nr_fp) if (nr_tp + nr_fp) > 0 else 0.0
    nr_recall = nr_tp / (nr_tp + nr_fn) if (nr_tp + nr_fn) > 0 else 0.0

    return {
        "n_samples": len(sample_records),
        "overall_field_accuracy": overall_accuracy,
        "field_accuracy": field_accuracy,
        "needs_review": {
            "precision": nr_precision,
            "recall": nr_recall,
            "tp": nr_tp,
            "fp": nr_fp,
            "fn": nr_fn,
        },
        "samples": sample_records,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on labeled val set")
    ap.add_argument("--val-jsonl", required=True, help="Path to val.jsonl with target_json labels")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--lora-adapter", default="",
                    help="Path to LoRA adapter directory (skip for zero-shot baseline)")
    ap.add_argument("--output", default="eval_results.json", help="Where to write full results JSON")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load samples
    raw_samples: list[dict] = []
    for line in Path(args.val_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw_samples.append(json.loads(line))
    labeled = [s for s in raw_samples if s.get("target_json")]
    print(f"Loaded {len(labeled)} labeled samples (out of {len(raw_samples)} total)")

    if not labeled:
        raise SystemExit(
            "No labeled samples found. Fill in 'target_json' in val.jsonl first."
        )

    # Load model
    print(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=1_003_520)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=None
    )
    model.to(device)

    if args.lora_adapter:
        print(f"Loading LoRA adapter: {args.lora_adapter}")
        model = PeftModel.from_pretrained(model, args.lora_adapter)

    model.eval()

    # Evaluate
    metrics = evaluate(model, processor, labeled, device, args.max_new_tokens)

    # Save full results
    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"Overall field accuracy:  {metrics['overall_field_accuracy']:.1%}")
    print(f"Samples evaluated:       {metrics['n_samples']}")
    print(f"\nPer-field accuracy:")
    for field, acc in metrics["field_accuracy"].items():
        bar = "█" * int(acc * 20)
        print(f"  {field:<30s} {acc:5.1%}  {bar}")
    nr = metrics["needs_review"]
    print(f"\nneeds_review  precision={nr['precision']:.2f}  recall={nr['recall']:.2f}"
          f"  (TP={nr['tp']} FP={nr['fp']} FN={nr['fn']})")
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
