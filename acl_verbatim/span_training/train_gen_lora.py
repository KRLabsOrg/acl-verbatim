import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for generative span extraction"
    )
    parser.add_argument("--train-file", required=True, help="Train JSONL")
    parser.add_argument("--eval-file", required=True, help="Eval JSONL")
    parser.add_argument(
        "--model",
        default="google/gemma-3-270m",
        help="HF model name",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def tokenize_row(tokenizer, row, max_length):
    prompt = row["prompt"]
    response = row["response"]
    full = prompt + response
    enc = tokenizer(
        full,
        truncation=True,
        max_length=max_length,
    )
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    labels = enc["input_ids"][:]
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    enc["labels"] = labels
    return enc


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_file, split="train")

    train_ds = train_ds.map(
        lambda r: tokenize_row(tokenizer, r, args.max_length),
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda r: tokenize_row(tokenizer, r, args.max_length),
        remove_columns=eval_ds.column_names,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        save_total_limit=2,
        seed=args.seed,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))


if __name__ == "__main__":
    main()
