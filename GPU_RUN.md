# MiniMind GPU 运行说明

仓库只提交代码和配置。训练数据、tokenizer、checkpoint、缓存和输出目录都已加入 `.gitignore`，需要在 GPU 服务器单独下载或上传。

## 在服务器准备环境

```bash
git clone <你的仓库地址> minimind
cd minimind
python -m pip install -r requirements.txt
mkdir -p dataset
```

自动下载本项目当前使用的预训练、SFT、LoRA、DPO 数据：

```bash
python scripts/prepare_data.py
```

默认先使用 ModelScope，失败后自动尝试 Hugging Face。也可以只下载某一组数据：

```bash
python scripts/prepare_data.py --files pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl
python scripts/prepare_data.py --source huggingface --files dpo.jsonl
```

官方数据说明见 README 中的 ModelScope 或 Hugging Face 链接。

如果服务器没有自定义 tokenizer，训练脚本会自动使用仓库中 `model/` 下的 tokenizer。需要复现实验中使用的 tokenizer 时，先执行：

```bash
cd trainer
python train_tokenizer.py
cd ..
```

## 先做 GPU 短跑

以下命令按仓库当前 768 hidden、8 层的配置运行，`max_steps` 只用于验证环境和显存，成功后删除它即可跑完整 epoch。

```bash
cd trainer

python train_pretrain.py \
  --device cuda:0 --dtype bfloat16 \
  --batch_size 4 --accumulation_steps 8 \
  --max_seq_len 768 --max_steps 100

python train_full_sft.py \
  --device cuda:0 --dtype bfloat16 \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain --batch_size 2 \
  --accumulation_steps 8 --max_seq_len 768 --max_steps 100

python train_lora.py \
  --device cuda:0 --dtype bfloat16 \
  --data_path ../dataset/lora_medical.jsonl \
  --from_weight full_sft --batch_size 4 \
  --accumulation_steps 8 --max_seq_len 768 --max_steps 100

python train_dpo.py \
  --device cuda:0 --dtype bfloat16 \
  --data_path ../dataset/dpo.jsonl \
  --from_weight full_sft --batch_size 2 \
  --accumulation_steps 8 --max_seq_len 768 --max_steps 100
```

输出会写入 `out/`，断点会写入 `checkpoints/`。48GB 显存足够运行这套 64M 级模型；如果显存利用率较低，可以逐步增大 `batch_size`，保持有效 batch size 基本不变。

## Git 提交和推送

在本地项目根目录执行：

```powershell
git switch -c gpu-ready
git add .gitignore GPU_RUN.md trainer scripts
git commit -m "Make MiniMind training portable for GPU"
git remote add mine <你的 Git 仓库地址>
git push -u mine gpu-ready
```

如果本地已经配置过自己的远程仓库，只需要把最后两条中的 `git remote add` 替换为对应的远程名和 `git push` 命令。服务器上以后更新代码：

```bash
git pull --ff-only
```
