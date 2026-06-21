"""YJH AI — 통합 CLI.

사용 (sovereign_ai/ 에서):
  python -m sovereign.cli acquire-arxiv     # arXiv 풀텍스트+메타 수집 (온라인)
  python -m sovereign.cli acquire-openalex  # OpenAlex 서지+초록 수집 (온라인)
  python -m sovereign.cli datasets          # 공개 데이터셋 다운로드 안내
  python -m sovereign.cli index             # 코퍼스 인덱싱 (오프라인 가능)
  python -m sovereign.cli ask "질문..."     # 1회 질의 (완전 오프라인)
  python -m sovereign.cli chat              # 대화형 (완전 오프라인)
  python -m sovereign.cli gui               # 데스크톱 창 GUI (완전 오프라인)
  python -m sovereign.cli status            # 현재 코퍼스/인덱스 상태
"""
from __future__ import annotations
import argparse
import json
import sys

from .config import load


def _print_sources(sources: list[dict]):
    if not sources:
        return
    print("\n── 근거 출처 ──")
    for n, h in enumerate(sources, 1):
        tag = "본문" if h.get("fulltext") else "초록"
        url = h.get("abs_url", "")
        print(f"[{n}] ({tag}, score={h.get('score',0):.3f}) {h.get('title','')[:90]}")
        if url:
            print(f"      {url}")


def cmd_acquire_arxiv(cfg, args):
    from .acquire import arxiv_harvest
    print(json.dumps(arxiv_harvest.harvest(cfg), ensure_ascii=False, indent=2))


def cmd_acquire_openalex(cfg, args):
    from .acquire import openalex_harvest
    print(json.dumps(openalex_harvest.harvest(cfg), ensure_ascii=False, indent=2))


def cmd_datasets(cfg, args):
    print("공개 데이터셋 (각 라이선스 준수, data/ 하위에 직접 다운로드):\n")
    for d in cfg.datasets:
        print(f"  • [{d['domain']}] {d['name']}\n      {d['url']}")


def cmd_index(cfg, args):
    from .ingest import build_index
    budget = getattr(args, "budget", None)
    print(json.dumps(build_index.build(cfg, time_budget_sec=budget),
                     ensure_ascii=False, indent=2))


def cmd_distill_gen(cfg, args):
    from .distill import generate_sft
    n = getattr(args, "n", None) or cfg.raw.get("distill", {}).get("n_examples", 200)
    teacher = getattr(args, "teacher", None)
    print(json.dumps(generate_sft.generate(cfg, n_examples=n, teacher_model=teacher),
                     ensure_ascii=False, indent=2))


def cmd_ask(cfg, args):
    from .serve.rag import RagEngine
    eng = RagEngine(cfg)
    res = eng.ask(args.query)
    print("\n" + res["answer"])
    _print_sources(res["sources"])


def cmd_chat(cfg, args):
    from .serve.rag import RagEngine
    eng = RagEngine(cfg)
    print("대화형 모드 (완전 오프라인). 종료: exit / quit / 빈 줄\n")
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q in ("exit", "quit"):
            break
        res = eng.ask(q)
        print("\n" + res["answer"])
        _print_sources(res["sources"])
        print()


def cmd_gui(cfg, args):
    # PyQt6는 GUI 실행 시에만 필요 → 지연 임포트 (다른 명령은 의존성 불필요)
    from .serve.gui import run
    run(cfg)


def cmd_status(cfg, args):
    cfg.ensure_dirs()
    def count_lines(p):
        return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
    print("── 코퍼스/인덱스 상태 ──")
    print(f"PDF(arxiv) : {len(list(cfg.pdf_dir.glob('*.pdf')))}개")
    print(f"PDF(보유)  : {len(list(cfg.user_pdf_dir.glob('*.pdf')))}개")
    print(f"arXiv 메타 : {count_lines(cfg.meta_dir / 'arxiv.jsonl')}건")
    print(f"OpenAlex   : {count_lines(cfg.meta_dir / 'openalex.jsonl')}건")
    info = cfg.index_dir / "info.json"
    if info.exists():
        print("인덱스     :", info.read_text(encoding="utf-8").strip())
    else:
        print("인덱스     : 없음 (index 명령 필요)")


COMMANDS = {
    "acquire-arxiv": cmd_acquire_arxiv,
    "acquire-openalex": cmd_acquire_openalex,
    "datasets": cmd_datasets,
    "index": cmd_index,
    "distill-gen": cmd_distill_gen,
    "ask": cmd_ask,
    "chat": cmd_chat,
    "gui": cmd_gui,
    "status": cmd_status,
}


def main(argv=None):
    # Windows 콘솔(cp949)에서 한글·기호 출력 깨짐/크래시 방지
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="sovereign", description="YJH AI (오프라인 RAG)")
    p.add_argument("--config", default=None, help="config.yaml 경로")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        if name == "ask":
            sp.add_argument("query", help="질문 문장")
        if name == "index":
            sp.add_argument("--budget", type=float, default=None,
                            help="이번 실행 시간예산(초). 초과분은 다음 실행에 이어서 처리")
        if name == "distill-gen":
            sp.add_argument("--n", type=int, default=None, help="생성할 (지시,답변) 쌍 수")
            sp.add_argument("--teacher", default=None, help="교사 모델(Ollama). 미지정시 config 값")
    args = p.parse_args(argv)
    cfg = load(args.config)
    COMMANDS[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
