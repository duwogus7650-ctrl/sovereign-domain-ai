"""도메인 질의 배치 실행 — 결과를 파일에 즉시 기록(버퍼링/크래시 영향 없음)."""
import traceback
from sovereign.config import load
from sovereign.serve.rag import RagEngine

OUT = "data/answers.txt"
qs = [
    "PMSM 센서리스 제어에서 저속 영역의 문제와 대표적 해법을 설명해줘.",
    "SiC 기반 인버터에서 게이트 드라이버를 설계할 때 핵심 고려사항은?",
    "강화학습을 전력전자나 모터 제어에 적용할 때의 주요 과제는 무엇인가?",
]

with open(OUT, "w", encoding="utf-8") as f:
    def w(s=""):
        f.write(s + "\n"); f.flush()
    try:
        eng = RagEngine(load())
        w("엔진 로드 완료\n")
        for q in qs:
            w("########################################")
            w("Q: " + q)
            w("----------------------------------------")
            try:
                res = eng.ask(q, top_k=5)
                w(res["answer"])
                w("--- 근거 ---")
                for n, h in enumerate(res["sources"], 1):
                    tag = "본문" if h.get("fulltext") else "초록"
                    src = h.get("venue") or h.get("source", "")
                    w(f"[{n}] ({tag}, {src}) " + h.get("title", "")[:65])
            except Exception:
                w("[질의 실패]\n" + traceback.format_exc())
            w("")
        w("=== DONE ===")
    except Exception:
        w("[엔진 로드 실패]\n" + traceback.format_exc())
