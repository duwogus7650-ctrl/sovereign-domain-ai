"""RAG 서빙 — 검색(FAISS) + 로컬 LLM(Ollama) 생성, 출처 인용 포함.

완전 오프라인: 인덱스와 Ollama 로컬 서버만 있으면 인터넷 불필요.
환각 억제: '근거에 없으면 모른다고 답하라'는 시스템 지시 + 출처 [n] 강제.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import requests

from ..config import Config

SYSTEM_PROMPT = """당신은 모터설계·제어·제어보드설계·AI고장진단·강화학습 분야의 오프라인 도메인 전문가다.
아래 제공된 '근거 자료'만을 사용해 한국어로 정확히 답하라.
규칙:
- 근거에 있는 내용만 말한다. 근거에 없으면 "제공된 자료에는 해당 내용이 없습니다"라고 분명히 밝힌다.
- 수식·수치는 근거에 있는 그대로만 인용한다. 임의로 만들지 않는다.
- 사용한 근거를 문장 끝에 [1], [2] 형태로 표시한다.
"""


class RagEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._index = None
        self._chunks: list[dict] = []
        self._titles: dict[str, str] = {}
        self._load_index()

    def _load_index(self):
        import faiss
        idx_path = self.cfg.index_dir / "faiss.index"
        ck_path = self.cfg.index_dir / "chunks.jsonl"
        if not idx_path.exists() or not ck_path.exists():
            raise FileNotFoundError(
                "인덱스가 없습니다. 먼저 `python -m sovereign.cli index`를 실행하세요.")
        # 한글 경로 호환: 바이트로 읽어 역직렬화 (build_index와 대칭)
        data = np.frombuffer(idx_path.read_bytes(), dtype="uint8")
        self._index = faiss.deserialize_index(data)
        with open(ck_path, "r", encoding="utf-8") as f:
            self._chunks = [json.loads(l) for l in f]
        self._load_titles()

    def _load_titles(self):
        """메타에서 doc_id→실제 제목 맵. PDF 청크(doc_id=arxiv_<id>)에 논문 제목을 붙임."""
        for mf in self.cfg.meta_dir.glob("*.jsonl"):
            with open(mf, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    aid, title = r.get("arxiv_id"), r.get("title", "")
                    if aid and title:
                        self._titles[f"arxiv_{aid}"] = title  # PDF stem 형식과 일치

    def _title_of(self, ch: dict) -> str:
        t = self._titles.get(ch.get("doc_id", ""))
        return t or ch.get("title", "")

    def _embed_query(self, q: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            dev = self.cfg.embedding.get("device", "auto")
            if dev == "auto":
                try:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    dev = "cpu"
            self._model = SentenceTransformer(self.cfg.embedding["model"], device=dev)
        return self._model.encode([q], normalize_embeddings=True).astype("float32")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or int(self.cfg.retrieval["top_k"])
        qv = self._embed_query(query)
        # 출처 다양성: 후보를 넉넉히 뽑아 문서(doc_id)당 최고점 1청크만 남김
        scores, idxs = self._index.search(qv, k * 6)
        min_score = float(self.cfg.retrieval.get("min_score", 0.0))
        hits, seen = [], set()
        for score, i in zip(scores[0], idxs[0]):
            if i < 0 or score < min_score:
                continue
            ch = dict(self._chunks[i])
            doc = ch.get("doc_id", "")
            if doc in seen:
                continue
            seen.add(doc)
            ch["score"] = float(score)
            ch["title"] = self._title_of(ch)   # PDF도 실제 논문 제목으로
            hits.append(ch)
            if len(hits) >= k:
                break
        return hits

    def _build_context(self, hits: list[dict]) -> str:
        blocks = []
        for n, h in enumerate(hits, 1):
            tag = "본문" if h.get("fulltext") else "초록"
            src = h.get("venue") or h.get("source", "")
            blocks.append(f"[{n}] ({tag}, {src}) {h.get('title','')}\n{h.get('text','')}")
        return "\n\n".join(blocks)

    def ask(self, query: str, top_k: int | None = None) -> dict:
        hits = self.retrieve(query, top_k)
        if not hits:
            return {"answer": "제공된 자료에서 관련 근거를 찾지 못했습니다.", "sources": []}
        context = self._build_context(hits)
        llm = self.cfg.llm
        user_msg = f"# 근거 자료\n{context}\n\n# 질문\n{query}\n\n근거에 기반해 답하고 [n]으로 출처를 표시하라."
        payload = {
            "model": llm["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            # qwen3 등 사고형 모델: CPU에서 <think> 생성이 매우 느려 기본 비활성.
            # 사고 추론이 필요하면 config llm.think: true
            "think": llm.get("think", False),
            # 모델을 메모리에 유지 → 요청 사이 콜드로드(수십초) 방지. CPU 사용성에 중요.
            "keep_alive": llm.get("keep_alive", "30m"),
            "options": {"temperature": llm.get("temperature", 0.2),
                        "num_ctx": llm.get("num_ctx", 8192)},
        }
        try:
            r = requests.post(f"{llm['base_url']}/api/chat", json=payload,
                              timeout=llm.get("timeout", 600))
            r.raise_for_status()
            answer = r.json()["message"]["content"]
        except Exception as e:
            answer = (f"[LLM 호출 실패: {e}]\nOllama가 실행 중인지, '{llm['model']}'가 "
                      f"pull 되었는지 확인하세요.\n\n검색된 근거는 아래 sources에 있습니다.")
        return {"answer": answer, "sources": hits}
