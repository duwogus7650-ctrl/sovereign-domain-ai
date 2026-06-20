"""인덱싱 — PDF/초록 → 텍스트 → 청크 → 임베딩 → FAISS 인덱스.

소스 3종을 하나의 코퍼스로 합친다:
  1) data/pdf/      : arXiv 등에서 받은 풀텍스트 PDF
  2) data/my_papers/: 사용자가 합법 보유한 PDF (선택)
  3) data/meta/*.jsonl : arXiv·OpenAlex 초록 (풀텍스트 없는 Q1 논문 커버)
산출: data/index/faiss.index + data/index/chunks.jsonl (청크별 메타·출처)

★ 증분·재개 가능: 이미 처리한 문서(doc_id)는 건너뛰고 진행분을 디스크에 저장.
  죽어도 다시 실행하면 이어서 진행. time_budget_sec로 포그라운드 제한도 회피.
★ 정합성 보장: chunks.jsonl 과 faiss 인덱스를 항상 같은 시점(in-memory all_meta)에서
  함께 저장한다 → 인덱스 위치 i ↔ 메타 i 가 절대 어긋나지 않음.
★ 견고성: PDF 텍스트의 깨진 유니코드(서러게이트)·널바이트를 제거해 토크나이저 크래시 방지.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

from ..config import Config

MAX_DOC_CHARS = 60000  # PDF당 최대 처리 길이(앞 ~15쪽). 양·시간 폭주 방지.


def _sanitize(text: str) -> str:
    # 잘못된 유니코드(고립 서러게이트)·널바이트 제거 — Rust 토크나이저 크래시 방지
    if not isinstance(text, str):
        return ""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text.replace("\x00", " ")


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(_sanitize(text).split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(path))
        out, total = [], 0
        for p in reader.pages:
            t = p.extract_text() or ""
            out.append(t)
            total += len(t)
            if total >= MAX_DOC_CHARS:
                break
        return "\n".join(out)
    except Exception as e:
        print(f"  ! PDF 추출 실패 {path.name}: {e}")
        return ""


def _iter_documents(cfg: Config, skip: set):
    """(doc_id, title, text, meta) 스트림. skip에 있는 doc_id는 추출 없이 건너뜀(재개 효율)."""
    for pdf_dir in (cfg.pdf_dir, cfg.user_pdf_dir):
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            if pdf.stem in skip:
                continue
            text = _pdf_text(pdf)
            if len(text) < 200:
                continue
            yield (pdf.stem, pdf.stem, text,
                   {"source": "pdf", "file": pdf.name, "fulltext": True})

    for meta_file in cfg.meta_dir.glob("*.jsonl"):
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                abstract = rec.get("abstract", "")
                if len(abstract) < 80:
                    continue
                base = rec.get("arxiv_id") or rec.get("openalex_id") or rec.get("doi") or rec.get("title", "")[:40]
                doc_id = f"abs::{base}"
                if doc_id in skip:
                    continue
                yield (doc_id, rec.get("title", ""), abstract,
                       {"source": rec.get("source", "meta"),
                        "domain": rec.get("domain", ""),
                        "venue": rec.get("venue", ""),
                        "doi": rec.get("doi", ""),
                        "abs_url": rec.get("abs_url", "") or rec.get("oa_url", ""),
                        "fulltext": False})


def _resolve_device(emb_cfg: dict) -> str:
    device = emb_cfg.get("device", "auto")
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def build(cfg: Config, time_budget_sec: float | None = None, save_every_docs: int = 300) -> dict:
    import faiss
    from sentence_transformers import SentenceTransformer

    cfg.ensure_dirs()
    emb_cfg = cfg.embedding
    model_name = emb_cfg["model"]
    faiss_path = cfg.index_dir / "faiss.index"
    chunks_path = cfg.index_dir / "chunks.jsonl"
    info_path = cfg.index_dir / "info.json"

    # ── 재개 판정: 같은 모델이면 이어쓰기. chunks.jsonl을 index.ntotal로 잘라 정합성 강제 ──
    processed: set[str] = set()
    all_meta: list[dict] = []
    index = None
    if info_path.exists() and faiss_path.exists() and chunks_path.exists():
        try:
            prev = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        if prev.get("model") == model_name:
            index = faiss.deserialize_index(np.frombuffer(faiss_path.read_bytes(), dtype="uint8"))
            lines = chunks_path.read_text(encoding="utf-8").splitlines()
            for line in lines[:index.ntotal]:        # 인덱스 개수만큼만 신뢰(고아 청크 폐기)
                try:
                    m = json.loads(line)
                    all_meta.append(m)
                    processed.add(m["doc_id"])
                except Exception:
                    pass
            print(f"[index] 재개: 인덱스 {index.ntotal}청크 / 메타 {len(all_meta)} / 문서 {len(processed)}개")

    device = _resolve_device(emb_cfg)
    print(f"[index] 임베딩 모델 로드: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device)

    if index is None:
        try:
            dim = model.get_embedding_dimension()
        except AttributeError:
            dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        print(f"[index] 새 인덱스 시작 (dim={dim})")

    size = int(emb_cfg["max_chars_per_chunk"])
    overlap = int(emb_cfg["chunk_overlap"])
    batch = int(emb_cfg["batch_size"])

    def save():
        # faiss와 chunks.jsonl을 같은 in-memory 상태에서 함께 기록 → 항상 정합
        faiss_path.write_bytes(faiss.serialize_index(index).tobytes())
        with open(chunks_path, "w", encoding="utf-8") as f:
            for m in all_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        info_path.write_text(json.dumps(
            {"model": model_name, "dim": index.d, "n_chunks": index.ntotal},
            ensure_ascii=False, indent=2), encoding="utf-8")

    buf_text: list[str] = []
    buf_meta: list[dict] = []

    def flush():
        if not buf_text:
            return
        vecs = model.encode(buf_text, batch_size=batch, normalize_embeddings=True).astype("float32")
        index.add(vecs)
        all_meta.extend(buf_meta)   # index.add 직후 동시 반영 → ntotal == len(all_meta)
        buf_text.clear()
        buf_meta.clear()

    start = time.time()
    n_new_docs = 0
    stopped_early = False
    for doc_id, title, text, meta in _iter_documents(cfg, processed):
        processed.add(doc_id)
        for ci, ch in enumerate(_chunk(text, size, overlap)):
            buf_text.append(ch)
            m = dict(meta)
            m.update({"doc_id": doc_id, "title": title, "chunk_index": ci, "text": ch})
            buf_meta.append(m)
        n_new_docs += 1
        if len(buf_text) >= 512:
            flush()
        if n_new_docs % save_every_docs == 0:
            flush(); save()
            print(f"  ... 신규문서 {n_new_docs} / 누적청크 {index.ntotal} / {time.time()-start:.0f}s")
        if time_budget_sec and (time.time() - start) > time_budget_sec:
            stopped_early = True
            break

    flush()
    save()

    # 정합성 자기검증 — 실패하면 즉시 드러나게
    assert index.ntotal == len(all_meta), f"정합성 깨짐: index {index.ntotal} != meta {len(all_meta)}"

    result = {"new_docs": n_new_docs, "total_chunks": index.ntotal,
              "dim": index.d, "stopped_early": stopped_early}
    print(f"[index] {'중단(시간예산) — 재실행 시 이어서' if stopped_early else '완료'}: "
          f"신규문서 {n_new_docs}, 누적청크 {index.ntotal}")
    return result
