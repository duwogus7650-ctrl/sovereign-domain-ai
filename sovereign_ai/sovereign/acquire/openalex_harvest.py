"""OpenAlex 수집 — Q1 유료 저널 포함 '전 논문'의 서지정보 + 초록(합법).

OpenAlex는 2억+ 논문의 메타데이터를 CC0로 공개한다. 본문은 받지 않고
제목·초록·저자·DOI·인용수·OA여부만 색인 → 유료 Q1 논문도 "무슨 연구가 있는지"는 커버.
초록은 inverted index 형태라 복원 함수 포함.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import requests

from ..config import Config

API = "https://api.openalex.org/works"


def _reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _load_seen(meta_path: Path) -> set[str]:
    seen = set()
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["openalex_id"])
                except Exception:
                    pass
    return seen


def harvest(cfg: Config) -> dict:
    cfg.ensure_dirs()
    acq = cfg.acquire
    meta_path = cfg.meta_dir / "openalex.jsonl"
    seen = _load_seen(meta_path)
    per = int(acq.get("openalex_per_query", 100))
    delay = float(acq.get("request_delay_sec", 3.0))
    mailto = acq.get("openalex_mailto", "")

    n_new = 0
    session = requests.Session()
    with open(meta_path, "a", encoding="utf-8") as out:
        for dom_key, dom in cfg.domains.items():
            for query in dom.get("arxiv_queries", []):
                # 제목+초록 한정 검색(정밀) + 관련도 기본정렬. 전역 search는 본문까지 훑어
                # 주제 무관 고인용 고전을 끌어오므로 쓰지 않는다.
                params = {
                    "filter": f"title_and_abstract.search:{query}",
                    "per-page": min(per, 200),
                }
                if mailto:
                    params["mailto"] = mailto
                print(f"[openalex] {dom['label']} :: {query}")
                try:
                    resp = session.get(API, params=params, timeout=60)
                    data = resp.json()
                except Exception as e:
                    print(f"  ! 요청 실패: {e}")
                    time.sleep(delay)
                    continue

                for w in data.get("results", []):
                    oid = w.get("id", "")
                    if not oid or oid in seen:
                        continue
                    seen.add(oid)
                    host = (w.get("primary_location") or {}).get("source") or {}
                    rec = {
                        "source": "openalex",
                        "openalex_id": oid,
                        "domain": dom_key,
                        "domain_label": dom["label"],
                        "title": w.get("title") or "",
                        "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
                        "doi": w.get("doi") or "",
                        "year": w.get("publication_year"),
                        "cited_by_count": w.get("cited_by_count", 0),
                        "venue": host.get("display_name", ""),
                        "is_oa": (w.get("open_access") or {}).get("is_oa", False),
                        "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_new += 1
                out.flush()
                time.sleep(delay)

    print(f"[openalex] 완료: 신규 {n_new}건")
    return {"new_records": n_new, "meta_file": str(meta_path)}
