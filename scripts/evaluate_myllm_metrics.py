"""Evaluate MiniMind checkpoints with the metric conventions used by MyLLM."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


DEFAULT_DATA = ROOT / "dataset"
DEFAULT_OUT = ROOT / "out" / "minimind_myllm_metrics_64m.json"
MAX_NEW = 256
MAX_INPUT = 512 - MAX_NEW
MIN_NEW = 8
REPETITION_PENALTY = 1.08
NO_REPEAT_NGRAM = 3


def chars(value):
    return list("".join(str(value).split()))


def rouge1_char(reference, prediction):
    ref, pred = chars(reference), chars(prediction)
    if not ref or not pred:
        return 0.0
    overlap = sum((Counter(ref) & Counter(pred)).values())
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rouge_l_char(reference, prediction):
    ref, pred = chars(reference), chars(prediction)
    if not ref or not pred:
        return 0.0
    previous = [0] * (len(pred) + 1)
    for ref_char in ref:
        current = [0]
        for index, pred_char in enumerate(pred, 1):
            current.append(
                previous[index - 1] + 1
                if ref_char == pred_char
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def char_f1(reference, prediction):
    return rouge1_char(reference, prediction)


def repetition_rate(text):
    value = chars(text)
    if len(value) < 2:
        return 0.0
    bigrams = [tuple(value[index:index + 2]) for index in range(len(value) - 1)]
    repeated = sum(1 for index, gram in enumerate(bigrams) if gram in bigrams[:index])
    return repeated / len(bigrams)


def is_corrupt(text):
    return any(marker in str(text) for marker in ("\ufffd", "閿熸枻鎷", "闁", "瀹搞儰绗"))


def _conversation_to_row(conversations):
    if not conversations or conversations[-1].get("role") != "assistant":
        return None
    history = conversations[:-1]
    question = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in history
    ).strip()
    answer = str(conversations[-1].get("content", "")).strip()
    return {"instruction": question, "input": "", "output": answer} if question and answer else None


def load_eval_data(data_dir, limit, data_file=None):
    rows = []
    paths = [Path(data_file)] if data_file else sorted(Path(data_dir).glob("*.json"))
    for path in paths:
        if path.suffix == ".jsonl":
            # JSONL records are newline-delimited; split only on LF because
            # medical/exam text can contain other Unicode line separators.
            items = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("instruction") and item.get("output"):
                rows.append(item)
            elif item.get("conversations"):
                row = _conversation_to_row(item["conversations"])
                if row:
                    rows.append(row)
    cut = int(len(rows) * 0.9)
    return rows[cut:cut + limit]


def load_model(checkpoint, tokenizer, device, lora_path=None, use_moe=False):
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=use_moe)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = model.half().eval().to(device)
    if lora_path:
        apply_lora(model)
        load_lora(model, lora_path)
        model = model.half()
        model.eval()
    return model


@torch.inference_mode()
def generate(model, tokenizer, question, device):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, open_thinking=False
    )
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT
    )
    input_ids = encoded["input_ids"].to(device)
    generated = []
    past_key_values = None
    eos = False
    eos_id = tokenizer.eos_token_id

    for _ in range(MAX_NEW):
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        result = model(
            input_ids[:, past_len:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = result.logits[:, -1, :].float()[0]
        seen = torch.unique(input_ids[0])
        scores = logits[seen]
        logits[seen] = torch.where(
            scores > 0, scores / REPETITION_PENALTY, scores * REPETITION_PENALTY
        )

        if len(generated) >= NO_REPEAT_NGRAM - 1:
            prefix = tuple(generated[-(NO_REPEAT_NGRAM - 1):])
            for index in range(len(generated) - NO_REPEAT_NGRAM + 1):
                if tuple(generated[index:index + NO_REPEAT_NGRAM - 1]) == prefix:
                    logits[generated[index + NO_REPEAT_NGRAM - 1]] = -float("inf")

        if len(generated) < MIN_NEW and eos_id is not None:
            logits[eos_id] = -float("inf")
        next_token = logits.argmax().view(1, 1)
        token_id = next_token.item()
        if eos_id is not None and token_id == eos_id:
            eos = True
            break
        generated.append(token_id)
        input_ids = torch.cat([input_ids, next_token.to(device)], dim=1)
        past_key_values = result.past_key_values

    return tokenizer.decode(generated, skip_special_tokens=True).strip(), eos


def evaluate(name, checkpoint, data, tokenizer, device, lora_path=None, use_moe=False):
    model = load_model(checkpoint, tokenizer, device, lora_path=lora_path, use_moe=use_moe)
    sums = {key: 0.0 for key in ("rouge1", "rougeL", "char_f1", "repeat_rate", "answer_chars")}
    eos_count = truncation_count = empty_count = corrupt_count = 0
    rows = []
    for index, item in enumerate(data, 1):
        question = str(item["instruction"]).strip()
        context = str(item.get("input", "")).strip()
        if context:
            question = f"{question}\n{context}"
        prediction, eos = generate(model, tokenizer, question, device)
        reference = str(item["output"])
        values = {
            "rouge1": rouge1_char(reference, prediction),
            "rougeL": rouge_l_char(reference, prediction),
            "char_f1": char_f1(reference, prediction),
            "repeat_rate": repetition_rate(prediction),
            "answer_chars": len(chars(prediction)),
        }
        for key, value in values.items():
            sums[key] += value
        eos_count += int(eos)
        truncation_count += int(not eos)
        empty_count += int(not prediction)
        corrupt_count += int(is_corrupt(prediction))
        rows.append({"question": question, "reference": reference, "prediction": prediction, **values})
        if index % 10 == 0 or index == len(data):
            print(f"{name}: {index}/{len(data)}")

    count = len(data)
    result = {
        "model": name,
        "checkpoint": str(checkpoint),
        "samples": count,
        **{key: value / count for key, value in sums.items()},
        "truncation_rate": truncation_count / count,
        "eos_rate": eos_count / count,
        "empty_rate": empty_count / count,
        "corrupt_rate": corrupt_count / count,
        "results": rows,
    }
    del model
    torch.cuda.empty_cache()
    return result


def main():
    global MAX_NEW, MAX_INPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new", type=int, default=MAX_NEW)
    parser.add_argument("--use-moe", action="store_true", help="评测 198M-A64M MoE 权重")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    MAX_NEW = max(1, args.max_new)
    MAX_INPUT = 512 - MAX_NEW
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer_path = ROOT / "c_pth" / "model_learn_tokenizer"
    if not tokenizer_path.is_dir():
        tokenizer_path = ROOT / "model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data = load_eval_data(args.data_dir, args.limit, args.data_file)
    suffix = "_moe" if args.use_moe else ""
    if args.output is None:
        args.output = ROOT / "out" / f"minimind_myllm_metrics_{'moe' if args.use_moe else '64m'}.json"
    checkpoints = [
        ("Base", ROOT / "out" / f"pretrain_768{suffix}.pth", None),
        ("SFT", ROOT / "out" / f"full_sft_768{suffix}.pth", None),
        ("LoRA", ROOT / "out" / f"full_sft_768{suffix}.pth", ROOT / "out" / f"lora_medical_768{suffix}.pth"),
        ("DPO", ROOT / "out" / f"dpo_768{suffix}.pth", None),
    ]
    results = []
    for name, checkpoint, lora_path in checkpoints:
        if not checkpoint.exists() or (lora_path and not lora_path.exists()):
            print(f"skip {name}: missing checkpoint")
            continue
        print(f"Evaluating {name}: {checkpoint}")
        results.append(evaluate(name, checkpoint, data, tokenizer, device, lora_path, use_moe=args.use_moe))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"device": device, "data_dir": str(args.data_dir), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_keys = (
        "model", "samples", "rouge1", "rougeL", "char_f1", "repeat_rate",
        "truncation_rate", "eos_rate", "empty_rate", "corrupt_rate", "answer_chars",
    )
    print(json.dumps([{key: row[key] for key in summary_keys} for row in results], ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
