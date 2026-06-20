"""arXiv 수집 — 합법 풀텍스트(오픈액세스 프리프린트)의 메타데이터 + PDF.

arXiv export API(Atom)를 사용한다. 도메인별 쿼리×카테고리로 검색하고,
서지정보를 data/meta/arxiv.jsonl 에 적재, 옵션에 따라 PDF를 data/pdf/ 에 저장.
중복(arxiv_id)은 자동 스킵 → 재실행 안전(idempotent).
"""
from __future__ import annotations
import json
import time
import urllib.parse
from pathlib import Path

import requests
import feedparser

from ..config import Config

API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _build_search(query: str, categories: list[str]) -> str:
    # 따옴표 구문검색은 너무 좁다 → 단어들을 AND로 묶어 주제 검색.
    # (all:word1 AND all:word2 ...) AND (cat:A OR cat:B ...)
    words = [w for w in query.split() if len(w) > 1]
    q = " AND ".join(f"all:{w}" for w in words) if words else f"all:{query}"
    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        return f"({q}) AND ({cats})"
    return f"({q})"


def _load_seen(meta_path: Path) -> set[str]:
    seen = set()
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["arxiv_id"])
                except Exception:
                    pass
    return seen


def harvest(cfg: Config) -> dict:
    cfg.ensure_dirs()
    acq = cfg.acquire
    meta_path = cfg.meta_dir / "arxiv.jsonl"
    seen = _load_seen(meta_path)
    delay = float(acq.get("request_delay_sec", 3.0))
    max_n = int(acq.get("arxiv_max_per_query", 40))
    dl_pdf = bool(acq.get("arxiv_download_pdf", True))

    n_new, n_pdf = 0, 0
    session = requests.Session()
    session.headers.update({"User-Agent": "sovereign-ai/0.1 (research; contact: local)"})

    with open(meta_path, "a", encoding="utf-8") as out:
        for dom_key, dom in cfg.domains.items():
            cats = dom.get("arxiv_categories", [])
            for query in dom.get("arxiv_queries", []):
                search = _build_search(query, cats)
                params = {
                    "search_query": search,
                    "start": 0,
                    "max_results": max_n,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                }
                url = f"{API}?{urllib.parse.urlencode(params)}"
                print(f"[arxiv] {dom['label']} :: {query}")
                try:
                    resp = session.get(url, timeout=60)
                    feed = feedparser.parse(resp.content)
                except Exception as e:
                    print(f"  ! 요청 실패: {e}")
                    time.sleep(delay)
                    continue

                for entry in feed.entries:
                    aid = entry.id.split("/abs/")[-1]  # e.g. 2401.12345v1
                    base_id = aid.split("v")[0]
                    if base_id in seen:
                        continue
                    seen.add(base_id)
                    rec = {
                        "source": "arxiv",
                        "arxiv_id": base_id,
                        "domain": dom_key,
                        "domain_label": dom["label"],
                        "title": entry.get("title", "").replace("\n", " ").strip(),
                        "abstract": entry.get("summary", "").replace("\n", " ").strip(),
                        "authors": [a.name for a in entry.get("authors", [])],
                        "published": entry.get("published", ""),
                        "pdf_url": next((l.href for l in entry.get("links", [])
                                         if l.get("type") == "application/pdf"), ""),
                        "abs_url": entry.get("link", ""),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    n_new += 1

                    if dl_pdf and rec["pdf_url"]:
                        pdf_path = cfg.pdf_dir / f"arxiv_{base_id}.pdf"
                        if not pdf_path.exists():
                            try:
                                r = session.get(rec["pdf_url"], timeout=120)
                                if r.ok and r.content[:4] == b"%PDF":
                                    pdf_path.write_bytes(r.content)
                                    n_pdf += 1
                                time.sleep(delay)
                            except Exception as e:
                                print(f"  ! PDF 실패 {base_id}: {e}")
                time.sleep(delay)

    summary = {"new_records": n_new, "pdfs_downloaded": n_pdf, "meta_file": str(meta_path)}
    print(f"[arxiv] 완료: 신규 {n_new}건, PDF {n_pdf}건")
    return summary
