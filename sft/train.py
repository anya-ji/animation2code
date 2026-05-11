#!/usr/bin/env python3
"""
SFT for Qwen3-VL on anim2code video -> HTML.

Usage:
    python sft/train.py --method lora --config sft/configs/lora.yaml
    python sft/train.py --method full --config sft/configs/full.yaml
    python sft/train.py --method lora --config sft/configs/lora.yaml --smoke
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "full"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None,
                        help="Override output_dir from config")
    parser.add_argument("--data-dir", default=None,
                        help="Local dir with {split}/metadata.jsonl + videos. "
                             "If absent, downloaded from HF.")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Limit training examples (for smoke testing)")
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Run a single forward+backward step then exit")
    parser.add_argument("--report-to", default=None,
                        help="Override report_to from config (e.g. 'none' for smoke)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.report_to:
        cfg["report_to"] = [args.report_to] if args.report_to != "none" else "none"
    if os.environ.get("WANDB_DISABLED", "").lower() == "true":
        cfg["report_to"] = "none"

    # ----- imports deferred so --help works without GPU stack -----
    from datasets import disable_caching
    disable_caching()
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    # ----- data -----
    if args.data_dir:
        local_dir = Path(args.data_dir)
    else:
        print("Downloading anim2code/anim2code-dev train + val splits from HF")
        local_dir = Path(snapshot_download(
            repo_id="anim2code/anim2code-dev",
            repo_type="dataset",
            allow_patterns=["data/train/*", "data/val/*"],
            token=os.environ.get("HF_TOKEN"),
            cache_dir=os.environ.get("HF_HOME"),
        )) / "data"
    print(f"data dir: {local_dir}")

    from data import AnimSFTDataset, Collator, load_split
    train_ex = load_split(local_dir, "train")
    val_ex = load_split(local_dir, "val")
    print(f"train: {len(train_ex)} examples, val: {len(val_ex)} examples")
    if args.max_train:
        train_ex = train_ex[:args.max_train]
    if args.max_eval:
        val_ex = val_ex[:args.max_eval]
    train_ds = AnimSFTDataset(train_ex, fps=cfg.get("fps", 1.0), max_pixels=cfg.get("max_pixels", 112896))
    val_ds = AnimSFTDataset(val_ex, fps=cfg.get("fps", 1.0), max_pixels=cfg.get("max_pixels", 112896))

    # ----- model + processor -----
    model_id = cfg["model_id"]
    if cfg.get("use_liger_kernel", True):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
            apply_liger_kernel_to_qwen3_vl()
            print("Applied Liger fused CE + RMSNorm + RoPE kernels for Qwen3-VL")
        except Exception as e:
            print(f"Liger kernel apply failed (continuing without): {e}")
    print(f"Loading model: {model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Required for gradient checkpointing to flow grads through frozen embeds.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg["lora"]["target_modules"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        # Full FT. Freeze the vision encoder to keep memory in budget.
        if cfg.get("freeze_vision", True):
            for n, p in model.named_parameters():
                if "visual" in n.lower() or "vision" in n.lower() or "patch_embed" in n.lower():
                    p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Full FT trainable: {trainable/1e9:.2f}B / {total/1e9:.2f}B")

    # ----- collator -----
    collator = Collator(processor, max_text_len=cfg.get("max_text_len", 16384))

    # ----- training args -----
    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=cfg.get("weight_decay", 0.0),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=True,
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.get("save_total_limit", 4),
        save_only_model=cfg.get("save_only_model", False),
        report_to=cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("run_name", f"sft-{args.method}"),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        max_steps=1 if args.smoke else -1,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"] + "/final")
    print("DONE")


if __name__ == "__main__":
    main()
