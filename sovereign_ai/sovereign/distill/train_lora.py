"""학생 모델 LoRA SFT — 교사가 만든 domain_sft.jsonl 로 도메인 전이.

GPU 권장(없으면 매우 느림). 학생 base는 오픈웨이트 소형 모델 또는
당신의 from-scratch 모델 디렉터리. 의존성:
  pip install transformers peft datasets accelerate trl
실행:
  python sovereign/distill/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct \
         --data data/sft/domain_sft.jsonl --out data/student_lora
self-distillation으로 가려면 --base 를 당신의 from-scratch 모델 경로로 바꾸면 된다.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def format_example(ex: dict) -> dict:
    # 간단한 chat 템플릿 (instruction → output)
    text = (f"<|user|>\n{ex['instruction']}\n<|assistant|>\n{ex['output']}")
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="학생 base 모델 (HF id 또는 from-scratch 모델 경로)")
    ap.add_argument("--data", default="data/sft/domain_sft.jsonl")
    ap.add_argument("--out", default="data/student_lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    print(f"[train] base={args.base}  data={args.data}")
    rows = [format_example(json.loads(l)) for l in open(args.data, encoding="utf-8")]
    tmp = Path(args.out); tmp.mkdir(parents=True, exist_ok=True)
    jl = tmp / "_train.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ds = load_dataset("json", data_files=str(jl), split="train")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    sft_cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, learning_rate=args.lr,
        max_seq_length=args.max_len, logging_steps=10, save_strategy="epoch",
        bf16=torch.cuda.is_available(), dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds,
                         peft_config=peft_cfg, tokenizer=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[train] 완료 → {args.out}")
    print("다음: LoRA 병합 → GGUF 변환(llama.cpp) → `ollama create`로 등록 → RAG가 그 모델을 호출")


if __name__ == "__main__":
    main()
