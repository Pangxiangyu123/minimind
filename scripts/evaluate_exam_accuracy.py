"""Evaluate multiple-choice accuracy on MiniMind's exam-format data."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from evaluate_myllm_metrics import load_eval_data, load_model, generate


def extract_choice(text):
    match = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", str(text).upper())
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, default=ROOT / "dataset" / "lora_exam.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "minimind_exam_metrics_64m.json")
    args = parser.parse_args()

    import evaluate_myllm_metrics as metrics
    metrics.MAX_NEW = 8
    metrics.MAX_INPUT = 504
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "c_pth" / "model_learn_tokenizer")
    data = load_eval_data(ROOT / "dataset", args.limit, args.data_file)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoints = [
        ("Base", ROOT / "out" / "pretrain_768.pth", None),
        ("SFT", ROOT / "out" / "full_sft_fixed_768.pth", None),
        ("LoRA", ROOT / "out" / "full_sft_fixed_768.pth", ROOT / "out" / "lora_medical_fixed_768.pth"),
        ("DPO", ROOT / "out" / "dpo_fixed_768.pth", None),
    ]
    results = []
    for name, checkpoint, lora_path in checkpoints:
        print(f"Evaluating {name}")
        model = load_model(checkpoint, tokenizer, device, lora_path=lora_path)
        correct = valid = eos_count = 0
        answer_chars = 0
        rows = []
        for index, item in enumerate(data, 1):
            prediction, eos = generate(model, tokenizer, item["instruction"], device)
            predicted_choice = extract_choice(prediction)
            reference_choice = extract_choice(item["output"])
            valid += int(predicted_choice is not None)
            correct += int(predicted_choice == reference_choice)
            eos_count += int(eos)
            answer_chars += len("".join(prediction.split()))
            rows.append({
                "question": item["instruction"],
                "reference": item["output"],
                "prediction": prediction,
                "reference_choice": reference_choice,
                "prediction_choice": predicted_choice,
            })
            if index % 20 == 0 or index == len(data):
                print(f"{name}: {index}/{len(data)}")
        n = len(data)
        results.append({
            "model": name,
            "samples": n,
            "accuracy": correct / n,
            "valid_choice_rate": valid / n,
            "eos_rate": eos_count / n,
            "answer_chars": answer_chars / n,
            "results": rows,
        })
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([
        {key: row[key] for key in ("model", "samples", "accuracy", "valid_choice_rate", "eos_rate", "answer_chars")}
        for row in results
    ], ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
