from __future__ import annotations

"""
LoRA fine-tuning for Qwen2.5-VL using QA pairs (multi-image input → nested JSON).

Key differences vs. the original train_lora.py:
- Each training sample contains MULTIPLE page images (full document) instead of
  a single page image. All images are packed into one user message.
- The target JSON is nested (budynki list, media list) rather than flat.
- Default max_seq_len is higher (6144) to accommodate multi-image sequences.
- Default max_pixels_per_image is lower (501_760 ≈ 7×reduction) to keep token
  counts manageable when 4-6 images are passed simultaneously.

Usage:
    source finetune_qwen_vl_pytorch/.venv311/bin/activate

    python finetune_qwen_vl_qa/train/train_lora_qa.py \
        --train-jsonl finetune_qwen_vl_qa/data/train.jsonl \
        --val-jsonl   finetune_qwen_vl_qa/data/val.jsonl \
        --out-dir     finetune_qwen_vl_qa/out/lora_run1 \
        [--model      Qwen/Qwen2.5-VL-7B-Instruct] \
        [--epochs     2] \
        [--lr         5e-5] \
        [--lora-r     16] \
        [--lora-alpha 32] \
        [--grad-accum 8] \
        [--max-seq-len 6144]
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from qa_utils import build_qa_prompt

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class QASample:
    pdf_stem: str
    image_paths: list[str]
    prompt: str
    target_text: str  # JSON string


def load_jsonl(path: Path) -> list[QASample]:
    samples: list[QASample] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        target = obj.get("target_json", {})
        if not target:
            continue  # skip unlabeled
        if not isinstance(target, str):
            target = json.dumps(target, ensure_ascii=False)
        img_paths = obj.get("image_paths", [])
        if not img_paths:
            continue
        samples.append(QASample(
            pdf_stem=obj.get("pdf_stem", ""),
            image_paths=img_paths,
            prompt=obj.get("prompt", build_qa_prompt()),
            target_text=target,
        ))
    return samples


# ---------------------------------------------------------------------------
# Batch / label building
# ---------------------------------------------------------------------------

def _user_msg(sample: QASample) -> dict[str, Any]:
    """Build the multi-image user message for one document."""
    content: list[dict[str, Any]] = []
    for img_path in sample.image_paths:
        abs_path = str(Path(img_path).expanduser().resolve())
        content.append({"type": "image", "image": f"file://{abs_path}"})
    content.append({"type": "text", "text": sample.prompt})
    return {"role": "user", "content": content}


def build_inputs_and_labels(
    processor: Any,
    sample: QASample,
    device: str,
    max_seq_len: int,
) -> dict[str, torch.Tensor] | None:
    """
    Build model inputs with correct causal-LM label masking.

    Same strategy as original: tokenise full conversation and prompt-only,
    mask everything up to (but not including) the assistant response.
    """
    user_msg = _user_msg(sample)
    asst_msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": sample.target_text}],
    }

    full_text = processor.apply_chat_template(
        [user_msg, asst_msg], tokenize=False, add_generation_prompt=False
    )
    prompt_text = processor.apply_chat_template(
        [user_msg], tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info([user_msg])
    video_inputs = video_inputs if video_inputs else None

    try:
        full_inputs = processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
    except Exception as exc:
        # Oversized image batch or other processor error
        print(f"  processor error for {sample.pdf_stem}: {exc}")
        return None

    full_len: int = full_inputs["input_ids"].shape[1]
    if full_len > max_seq_len:
        return None  # skip; too long

    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )
    prompt_len: int = prompt_inputs["input_ids"].shape[1]
    response_len: int = full_len - prompt_len

    if response_len <= 0:
        return None

    labels = full_inputs["input_ids"].clone()
    labels[:, :-response_len] = IGNORE_INDEX

    result = {k: v.to(device) for k, v in full_inputs.items() if hasattr(v, "to")}
    result["labels"] = labels.to(device)
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_val(model, processor, val_samples, device, max_seq_len) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for sample in tqdm(val_samples, desc="  val", leave=False):
            batch = build_inputs_and_labels(processor, sample, device, max_seq_len)
            if batch is None:
                continue
            out = model(**batch)
            total_loss += float(out.loss.detach().cpu())
            n += 1
            if device == "mps":
                torch.mps.empty_cache()
    model.train()
    return total_loss / max(1, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="QA-pair LoRA fine-tune Qwen2.5-VL (multi-image)")
    p.add_argument("--model",           default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--train-jsonl",     required=True)
    p.add_argument("--val-jsonl",       default="")
    p.add_argument("--out-dir",         required=True)
    p.add_argument("--epochs",          type=int,   default=2)
    p.add_argument("--lr",              type=float, default=5e-5)
    p.add_argument("--warmup-steps",    type=int,   default=50)
    p.add_argument("--grad-accum",      type=int,   default=8,
                   help="Effective batch = grad_accum × 1 (reduced vs original because "
                        "each sample now contains multiple images)")
    p.add_argument("--lora-r",          type=int,   default=16)
    p.add_argument("--lora-alpha",      type=int,   default=32)
    p.add_argument("--lora-dropout",    type=float, default=0.05)
    p.add_argument("--max-seq-len",     type=int,   default=6144,
                   help="Skip documents whose full tokenised length exceeds this. "
                        "Multi-image documents tokenise to ~3000-5000 tokens at 200 DPI.")
    p.add_argument("--max-pixels",      type=int,   default=501_760,
                   help="max_pixels per image for the processor (default 501_760 = ~512 tokens/image). "
                        "Reduce to 250_880 if you hit OOM with 6+ pages.")
    p.add_argument("--save-every",      type=int,   default=50,
                   help="Save checkpoint every N optimizer steps (0 = off). "
                        "Lower default because document-level steps are fewer than page-level.")
    p.add_argument("--seed",            type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=None,
    )
    model.to(device)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    train_samples = load_jsonl(Path(args.train_jsonl))
    val_samples   = load_jsonl(Path(args.val_jsonl)) if args.val_jsonl else []
    random.shuffle(train_samples)
    print(f"Train: {len(train_samples)} documents  Val: {len(val_samples)} documents")

    total_opt_steps = math.ceil(len(train_samples) / args.grad_accum) * args.epochs
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_opt_steps,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    opt_step, micro_step, skipped = 0, 0, 0
    optim.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(train_samples, desc=f"epoch {epoch + 1}/{args.epochs}")
        for sample in pbar:
            batch = build_inputs_and_labels(processor, sample, device, args.max_seq_len)
            if batch is None:
                skipped += 1
                continue

            out = model(**batch)
            loss = out.loss / float(args.grad_accum)
            loss.backward()
            micro_step += 1

            if device == "mps":
                torch.mps.empty_cache()

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1

                pbar.set_postfix({
                    "loss": f"{float(out.loss.detach().cpu()):.4f}",
                    "lr":   f"{sched.get_last_lr()[0]:.1e}",
                    "step": opt_step,
                })

                if args.save_every > 0 and opt_step % args.save_every == 0:
                    ckpt = out_dir / f"checkpoint-{opt_step}"
                    model.save_pretrained(str(ckpt))
                    processor.save_pretrained(str(ckpt))
                    print(f"\n  Saved checkpoint: {ckpt}")

        # Flush remaining gradient
        if micro_step % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            opt_step += 1

        if val_samples:
            val_loss = run_val(model, processor, val_samples, device, args.max_seq_len)
            print(f"  → epoch {epoch + 1} val loss: {val_loss:.4f}")

    if skipped:
        print(f"Skipped {skipped} samples (too long or empty target).")

    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    print(f"\nSaved final LoRA adapter + processor to: {out_dir}")


if __name__ == "__main__":
    main()
