"""Evaluate chosen/rejected preference margins on DPO holdout data."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import DPODataset
from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def sequence_logprob(model, x, y, mask):
    logits = model(x).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, y.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=1), mask.sum(dim=1).clamp_min(1)


def load_model(checkpoint, device, lora_path=None):
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = model.half().eval().to(device)
    if lora_path:
        apply_lora(model)
        load_lora(model, lora_path)
        model = model.half()
        model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, default=ROOT / "dataset" / "dpo.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=368)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "minimind_dpo_metrics_64m.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "c_pth" / "model_learn_tokenizer")
    full_dataset = DPODataset(str(args.data_file), tokenizer, max_length=args.max_seq_len)
    start = int(len(full_dataset) * 0.9)
    indexes = list(range(start, min(start + args.limit, len(full_dataset))))
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
        model = load_model(checkpoint, device, lora_path=lora_path)
        chosen_scores = []
        rejected_scores = []
        chosen_counts = []
        rejected_counts = []
        with torch.inference_mode():
            for offset in range(0, len(indexes), args.batch_size):
                batch = [full_dataset[index] for index in indexes[offset:offset + args.batch_size]]
                chosen, chosen_count = sequence_logprob(
                    model,
                    torch.stack([row["x_chosen"] for row in batch]).to(device),
                    torch.stack([row["y_chosen"] for row in batch]).to(device),
                    torch.stack([row["mask_chosen"] for row in batch]).to(device),
                )
                rejected, rejected_count = sequence_logprob(
                    model,
                    torch.stack([row["x_rejected"] for row in batch]).to(device),
                    torch.stack([row["y_rejected"] for row in batch]).to(device),
                    torch.stack([row["mask_rejected"] for row in batch]).to(device),
                )
                chosen_scores.extend(chosen.cpu().tolist())
                rejected_scores.extend(rejected.cpu().tolist())
                chosen_counts.extend(chosen_count.cpu().tolist())
                rejected_counts.extend(rejected_count.cpu().tolist())
        margins = [chosen - rejected for chosen, rejected in zip(chosen_scores, rejected_scores)]
        chosen_avg = [score / count for score, count in zip(chosen_scores, chosen_counts)]
        rejected_avg = [score / count for score, count in zip(rejected_scores, rejected_counts)]
        avg_margins = [chosen - rejected for chosen, rejected in zip(chosen_avg, rejected_avg)]
        n = len(margins)
        results.append({
            "model": name,
            "samples": n,
            "preference_accuracy": sum(margin > 0 for margin in margins) / n,
            "token_normalized_preference_accuracy": sum(margin > 0 for margin in avg_margins) / n,
            "mean_margin": sum(margins) / n,
            "mean_token_normalized_margin": sum(avg_margins) / n,
            "mean_chosen_logprob": sum(chosen_scores) / n,
            "mean_rejected_logprob": sum(rejected_scores) / n,
            "mean_chosen_token_logprob": sum(chosen_avg) / n,
            "mean_rejected_token_logprob": sum(rejected_avg) / n,
        })
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
