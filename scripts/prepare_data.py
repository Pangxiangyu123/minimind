"""Download the MiniMind datasets needed by the local training scripts.

The files are intentionally kept outside Git.  ModelScope is the default
source for GPU servers in China; Hugging Face is available as a fallback.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "dataset"
DATASET_REPO = "gongjy/minimind_dataset"
HF_DATASET_REPO = "jingyaogong/minimind_dataset"
SUPPORTED_FILES = [
    "agent_rl.jsonl",
    "agent_rl_math.jsonl",
    "dpo.jsonl",
    "lora_exam.jsonl",
    "lora_identity.jsonl",
    "lora_medical.jsonl",
    "pretrain_t2t.jsonl",
    "pretrain_t2t_mini.jsonl",
    "rlaif.jsonl",
    "sft_t2t.jsonl",
    "sft_t2t_mini.jsonl",
]
DEFAULT_FILES = [
    "pretrain_t2t_mini.jsonl",
    "sft_t2t_mini.jsonl",
    "lora_medical.jsonl",
    "lora_exam.jsonl",
    "lora_identity.jsonl",
    "dpo.jsonl",
]


def _copy_to_target(source: str | Path, target: Path) -> None:
    source = Path(source)
    if source.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def download_modelscope(filename: str, data_dir: Path, revision: str) -> Path:
    from modelscope.hub.file_download import dataset_file_download

    downloaded = dataset_file_download(
        dataset_id=DATASET_REPO,
        file_path=filename,
        revision=revision,
        local_dir=str(data_dir),
    )
    return Path(downloaded)


def download_huggingface(filename: str, data_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=filename,
        local_dir=str(data_dir),
    )
    return Path(downloaded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MiniMind training datasets")
    parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "huggingface"),
        default="auto",
        help="数据源；auto 先尝试 ModelScope，失败后尝试 Hugging Face",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--force", action="store_true", help="重新下载已存在的文件")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES, choices=SUPPORTED_FILES)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for filename in args.files:
        target = args.data_dir / filename
        if target.is_file() and target.stat().st_size > 0 and not args.force:
            print(f"[skip] {target}")
            continue

        sources = [args.source] if args.source != "auto" else ["modelscope", "huggingface"]
        downloaded = False
        for source in sources:
            try:
                print(f"[download:{source}] {filename}")
                if source == "modelscope":
                    path = download_modelscope(filename, args.data_dir, args.revision)
                else:
                    path = download_huggingface(filename, args.data_dir)
                _copy_to_target(path, target)
                print(f"[ready] {target} ({target.stat().st_size / 1024**2:.1f} MiB)")
                downloaded = True
                break
            except Exception as exc:
                print(f"[failed:{source}] {filename}: {exc}")

        if not downloaded:
            failed.append(filename)

    if failed:
        raise SystemExit(f"无法下载: {', '.join(failed)}")


if __name__ == "__main__":
    main()
