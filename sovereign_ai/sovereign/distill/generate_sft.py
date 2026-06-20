"""시퀀스레벨(블랙박스) 증류 — 오픈웨이트 교사로 도메인 SFT 데이터 생성.

폐쇄형 프런티어(Claude/GPT/Gemini)는 약관상 사용하지 않는다(docs/03 참고).
교사는 로컬 Ollama의 오픈웨이트 모델(예: qwen3:14b). 코퍼스 청크를 근거로 주고
(지시, 근거기반 답변) 쌍을 만들어 data/sft/domain_sft.jsonl 에 적재한다.
이 데이터로 학생을 LoRA SFT 하면 '도메인 추론 방식'이 학생에게 전이된다(사실은 RAG가 담당).
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import requests

from ..config import Config

GEN_SYSTEM = """당신은 모터·제어·전력전자·고장진단·강화학습 분야 교수다.
주어진 '근거 단락'을 바탕으로, 그 분야 학생을 가르치기 위한 고품질 학습용 질문-답변 1쌍을 만들어라.
- 질문은 근거로 답할 수 있는 구체적·기술적 질문일 것.
- 답변은 근거에 충실하고, 논리적이며, 한국어로 쓸 것.
- 반드시 아래 JSON만 출력: {"instruction": "...", "output": "..."}"""


def _ollama_chat(base_url: str, model: str, system: str, user: str,
                 temperature: float = 0.4, num_ctx: int = 8192) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def generate(cfg: Config, n_examples: int = 200, seed: int = 0) -> dict:
    dcfg = cfg.raw.get("distill", {})
    teacher = dcfg.get("teacher_model", "qwen3:14b")
    base_url = cfg.llm["base_url"]

    chunks_path = cfg.index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError("먼저 `index`로 코퍼스를 인덱싱하세요 (chunks.jsonl 필요).")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f]
    # 본문 청크 우선, 길이 충분한 것만
    pool = [c for c in chunks if len(c.get("text", "")) > 300]
    if not pool:
        pool = chunks
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[:n_examples]

    out_dir = cfg.data_root / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "domain_sft.jsonl"

    n_ok, n_fail = 0, 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, ch in enumerate(pool, 1):
            evidence = ch["text"][:1500]
            title = ch.get("title", "")
            user = f"근거 단락 (출처: {title}):\n{evidence}"
            try:
                raw = _ollama_chat(base_url, teacher, GEN_SYSTEM, user)
                obj = json.loads(raw)
                if "instruction" in obj and "output" in obj:
                    obj["_meta"] = {"domain": ch.get("domain", ""), "title": title}
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"  ! 생성 실패({i}): {e}")
            if i % 20 == 0:
                print(f"  ... {i}/{len(pool)} (성공 {n_ok})")

    print(f"[distill-gen] 완료: {n_ok}쌍 생성 → {out_path} (실패 {n_fail})")
    return {"examples": n_ok, "failed": n_fail, "out": str(out_path), "teacher": teacher}
