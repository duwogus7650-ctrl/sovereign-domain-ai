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
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--merge", action="store_true",
                    help="학습 후 LoRA를 base에 병합해 {out}/merged 에 풀 모델 저장(GGUF 변환용)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    print(f"[train] base={args.base}  data={args.data}")
    rows = [format_example(json.loads(l)) for l in open(args.data, encoding="utf-8")
            if l.strip()]
    print(f"[train] 학습 샘플 {len(rows)}개")
    tmp = Path(args.out); tmp.mkdir(parents=True, exist_ok=True)
    jl = tmp / "_train.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ds = load_dataset("json", data_files=str(jl), split="train")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16 if use_cuda else torch.float32)

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    sft_cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
        max_seq_length=args.max_len, logging_steps=10, save_strategy="epoch",
        bf16=use_cuda, dataset_text_field="text",
    )
    # trl 버전에 따라 tokenizer/processing_class 인자명이 다름 → 양쪽 시도
    try:
        trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds,
                             peft_config=peft_cfg, processing_class=tok)
    except TypeError:
        trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds,
                             peft_config=peft_cfg, tokenizer=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[train] LoRA 어댑터 저장 → {args.out}")

    if args.merge:
        from peft import PeftModel
        print("[train] LoRA를 base에 병합 중...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16 if use_cuda else torch.float32)
        merged = PeftModel.from_pretrained(base, args.out).merge_and_unload()
        merged_dir = str(Path(args.out) / "merged")
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"[train] 병합 풀 모델 저장 → {merged_dir}")

    print("다음: (병합 모델을) GGUF 변환(llama.cpp) → `ollama create`로 등록 "
          "→ config.yaml llm.model 을 그 모델로. 상세는 docs/04_증류_실행_런북.md")


if __name__ == "__main__":
    main()
