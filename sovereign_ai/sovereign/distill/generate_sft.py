"""시퀀스레벨(블랙박스) 증류 — 오픈웨이트 교사로 도메인 SFT 데이터 생성.

폐쇄형 프런티어(Claude/GPT/Gemini)는 약관상 사용하지 않는다(docs/03 참고).
교사는 로컬 Ollama의 오픈웨이트 모델(예: qwen3:8b, 더 크면 14b/32b). 코퍼스 청크를
근거로 주고 (지시, 근거기반 답변) 쌍을 만들어 data/sft/domain_sft.jsonl 에 적재한다.

★ 재개 가능(resumable): 같은 seed로 청크 풀을 정렬하므로, 중간에 멈춰도 다시 실행하면
  이미 생성한 개수만큼 건너뛰고 목표치까지 이어서 생성한다(append).
★ 견고성: 교사 출력이 깨진 JSON이어도 {...} 구간을 추출해 복구 시도, 실패는 건너뜀.
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path

import requests

from ..config import Config

GEN_SYSTEM = """당신은 모터·제어·전력전자·고장진단·강화학습 분야 교수다.
주어진 '근거 단락'만을 바탕으로, 그 분야 대학원생을 가르치기 위한 고품질 학습용 질문-답변 1쌍을 만들어라.
규칙:
- 질문(instruction)은 근거로 답할 수 있는 구체적·기술적 질문. 근거 단락을 그대로 베끼지 말 것.
- 답변(output)은 근거에 충실하고 논리적이며 한국어로 작성. 근거에 없는 수치는 지어내지 말 것.
- 출력은 오직 아래 JSON 한 줄: {"instruction": "...", "output": "..."}
- 다른 텍스트·설명·코드블록 표시 없이 JSON만."""


def _ollama_chat(base_url: str, model: str, system: str, user: str,
                 temperature: float = 0.4, num_ctx: int = 8192) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": "json",
        "think": False,   # 사고형 모델의 <think> 생성으로 인한 지연 방지
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _parse_pair(raw: str) -> dict | None:
    """교사 출력에서 {instruction, output} 추출. 깨진 JSON도 복구 시도."""
    for candidate in (raw, _extract_json(raw)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        instr, out = obj.get("instruction"), obj.get("output")
        if isinstance(instr, str) and isinstance(out, str) and len(instr) > 5 and len(out) > 20:
            return {"instruction": instr.strip(), "output": out.strip()}
    return None


def _extract_json(s: str) -> str | None:
    m = re.search(r"\{.*\}", s, re.DOTALL)
    return m.group(0) if m else None


def _load_pool(cfg: Config, seed: int) -> list[dict]:
    chunks_path = cfg.index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError("먼저 `index`로 코퍼스를 인덱싱하세요 (chunks.jsonl 필요).")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f]
    pool = [c for c in chunks if len(c.get("text", "")) > 300] or chunks
    rng = random.Random(seed)
    rng.shuffle(pool)            # seed 고정 → 재개 시 동일 순서
    return pool


def generate(cfg: Config, n_examples: int = 200, seed: int = 0,
             teacher_model: str | None = None) -> dict:
    dcfg = cfg.raw.get("distill", {})
    teacher = teacher_model or dcfg.get("teacher_model", "qwen3:8b")
    base_url = cfg.llm["base_url"]

    pool = _load_pool(cfg, seed)
    out_dir = cfg.data_root / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "domain_sft.jsonl"

    # 재개: 이미 생성된 줄 수만큼 건너뜀
    done = 0
    if out_path.exists():
        done = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
    if done >= n_examples:
        print(f"[distill-gen] 이미 {done}쌍 생성됨 (목표 {n_examples}). 더 만들려면 n을 늘리세요.")
        return {"examples": done, "failed": 0, "out": str(out_path), "teacher": teacher}
    print(f"[distill-gen] 교사={teacher} | 재개 {done}/{n_examples}쌍부터")

    targets = pool[done:n_examples]
    n_ok, n_fail = done, 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i, ch in enumerate(targets, start=done + 1):
            user = f"근거 단락 (출처: {ch.get('title','')}):\n{ch['text'][:1500]}"
            try:
                pair = _parse_pair(_ollama_chat(base_url, teacher, GEN_SYSTEM, user))
                if pair:
                    pair["_meta"] = {"domain": ch.get("domain", ""), "title": ch.get("title", "")}
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    out.flush()
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"  ! 생성 실패({i}): {e}")
            if i % 20 == 0:
                print(f"  ... {i}/{n_examples} (성공 누적 {n_ok}, 실패 {n_fail})")

    print(f"[distill-gen] 완료: 총 {n_ok}쌍 → {out_path} (이번 실패 {n_fail})")
    return {"examples": n_ok, "failed": n_fail, "out": str(out_path), "teacher": teacher}
