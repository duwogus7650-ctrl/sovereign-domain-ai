"""Grounding/faithfulness eval harness for the RAG engine.

Closes the 'grounding is prompt-only, nothing checks compliance' gap:
the SYSTEM_PROMPT tells the LLM to use only evidence and cite [n], but nothing
verified it. This module MEASURES it. Deterministic (lexical), stdlib-only, so
it runs without Ollama/FAISS/corpus. Self-test: `python eval_grounding.py`.

- retrieval_metrics(retrieve_fn, labeled, k): recall@k, MRR, nDCG vs labeled relevant doc_ids.
- citation_faithfulness(answer, sources): per claim-sentence, does its [n] citation
  actually support it? -> faithfulness, uncited-claims, invalid-citation counts.
Full GREEN still needs a real labeled query set + real LLM answers on the real corpus.
"""
from __future__ import annotations
import re, math

_TOK = re.compile(r"[0-9A-Za-z가-힣]+")
_CITE = re.compile(r"\[(\d+)\]")
_SENT = re.compile(r"[^.!?。\n]+[.!?。]?")

def _tok(s):
    return [t for t in _TOK.findall((s or "").lower()) if len(t) > 1]

def _support(claim, evidence):
    ct = _tok(claim)
    if not ct:
        return 1.0
    ev = set(_tok(evidence))
    return sum(1 for t in ct if t in ev) / len(ct)

def citation_faithfulness(answer, sources, support_thr=0.5,
                          refusal=("해당 내용이 없", "찾지 못", "자료에는", "없습니다")):
    src = [(s.get("text", "") if isinstance(s, dict) else str(s)) for s in sources]
    rows, claims, supported, uncited, invalid = [], 0, 0, 0, 0
    _ans = re.sub(r"(?<=\d)\.(?=\d)", "․", answer)  # protect decimal points from sentence split
    for sent in ((x.replace("․", ".")).strip() for x in _SENT.findall(_ans) if len(x.strip()) > 1):
        cites = [int(x) for x in _CITE.findall(sent)]
        if any(m in sent for m in refusal) or len(_tok(sent)) < 3:
            rows.append((sent, cites, None, "non-claim/refusal")); continue
        claims += 1
        bad = [c for c in cites if not (1 <= c <= len(src))]
        invalid += len(bad)
        valid = [c for c in cites if 1 <= c <= len(src)]
        if not cites:
            uncited += 1; rows.append((sent, cites, 0.0, "UNCITED claim")); continue
        sup = max((_support(sent, src[c-1]) for c in valid), default=0.0)
        if sup >= support_thr: supported += 1
        rows.append((sent, cites, round(sup, 2),
                     "supported" if sup >= support_thr else "unsupported/hallucinated"))
    return {"faithfulness": supported/claims if claims else 1.0,
            "claim_sentences": claims, "supported": supported,
            "uncited_claims": uncited, "invalid_citations": invalid, "rows": rows}

def retrieval_metrics(retrieve_fn, labeled, k=5):
    rec, rr, nd = [], [], []
    for it in labeled:
        docs = [h.get("doc_id") for h in retrieve_fn(it["query"], k)]
        rel = set(it["relevant"])
        flags = [1 if d in rel else 0 for d in docs]
        rec.append(sum(flags)/max(1, len(rel)))
        rr.append(next((1/i for i, f in enumerate(flags, 1) if f), 0.0))
        dcg = sum(f/math.log2(i+1) for i, f in enumerate(flags, 1))
        idcg = sum(1/math.log2(i+1) for i in range(1, min(len(rel), k)+1))
        nd.append(dcg/idcg if idcg else 0.0)
    n = len(labeled) or 1
    return {"recall@k": sum(rec)/n, "MRR": sum(rr)/n, "nDCG": sum(nd)/n, "n": len(labeled)}

if __name__ == "__main__":
    sources = [
        {"doc_id": "A", "text": "PMSM의 토크는 Te = 1.5 p lambda_pm iq 로 계산된다."},
        {"doc_id": "B", "text": "FOC 전류루프 이득은 Kp = L wc, Ki = R wc 로 설계한다."},
        {"doc_id": "C", "text": "베어링 외륜 결함 주파수는 BPFO 이다."},
    ]
    faithful = "PMSM 토크는 Te 1.5 p lambda_pm iq 로 계산된다 [1]. 전류루프 이득은 Kp L wc Ki R wc 로 설계한다 [2]."
    halluc = "PMSM 효율은 항상 99 퍼센트 이다 [1]. 최대 속도는 10000 rpm 이다 [2]. 수냉 냉각이 필수다 [5]."
    ff = citation_faithfulness(faithful, sources)
    hf = citation_faithfulness(halluc, sources)
    print("faithful answer :", {k: ff[k] for k in ("faithfulness","claim_sentences","supported","invalid_citations")})
    print("hallucinated    :", {k: hf[k] for k in ("faithfulness","claim_sentences","supported","invalid_citations")})
    labeled = [{"query": "PMSM 토크 계산", "relevant": {"A"}}, {"query": "FOC 전류루프 이득", "relevant": {"B"}}]
    fake = lambda q, k: ([{"doc_id":"A"},{"doc_id":"C"}] if "토크" in q else [{"doc_id":"B"},{"doc_id":"A"}])
    rm = retrieval_metrics(fake, labeled, k=2)
    print("retrieval       :", rm)
    assert ff["faithfulness"] > 0.9, ff
    assert hf["faithfulness"] < 0.4, hf
    assert hf["invalid_citations"] >= 1, "should catch out-of-range [5]"
    assert rm["recall@k"] == 1.0 and rm["MRR"] == 1.0
    print("\nSELFTEST PASS: distinguishes faithful(>0.9) vs hallucinated(<0.4), catches invalid [5], retrieval recall/MRR correct.")
