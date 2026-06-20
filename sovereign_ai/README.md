# 소버린 도메인 AI

모터설계 · 제어 · 제어보드설계 · AI고장진단 · 강화학습 — **오프라인 구동** 도메인 전문 AI.

당신의 노트(`../05·06`)가 내린 결론 위에 선다: **순수성 지향 + 방법 모방**. 거기에
"논문을 실제로 안다 + 오프라인"을 더하기 위해 **RAG 우선** 아키텍처를 채택했고,
"증류로 프런티어 모방" 요청은 **합법(오픈웨이트 교사) 시퀀스레벨 증류**로 통합했다.

## 한눈에 보는 구조

```
[수집] arXiv 풀텍스트 + OpenAlex 초록 + 공개 데이터셋   ← 온라인 1회
   │
[인덱싱] PDF/초록 → 청크 → bge-m3 임베딩 → FAISS         ← 이후 오프라인
   │
[서빙] 질문 → FAISS 검색 → 로컬 LLM(Ollama/Qwen3) 답변+출처  ← 완전 오프라인
   │
[증류] 오픈웨이트 교사 → 도메인 SFT 데이터 → 학생 LoRA      ← 선택, 도메인 말투 전이
```

## 빠른 시작

```bash
cd sovereign_ai
pip install -r requirements.txt
# 로컬 LLM (별도): https://ollama.com  →  ollama pull qwen3:8b

# 1) 수집 (온라인, config.yaml의 한도는 처음엔 작게)
python -m sovereign.cli acquire-arxiv
python -m sovereign.cli acquire-openalex
python -m sovereign.cli datasets        # 공개 데이터셋 다운로드 링크 출력

# 2) 인덱싱 (임베딩 모델 1회 다운로드 후 오프라인)
python -m sovereign.cli index

# 3) 질의 (완전 오프라인)
python -m sovereign.cli ask "PMSM 센서리스 제어에서 저속 영역 문제와 대표 해법은?"
python -m sovereign.cli chat
python -m sovereign.cli status

# 4) (선택) 증류: 오픈웨이트 교사로 도메인 학습데이터 생성 → 학생 LoRA
python -m sovereign.cli distill-gen
python sovereign/distill/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct
```

## 문서
- `docs/01_도메인맵_논문데이터.md` — 5개 도메인별 arXiv 카테고리 · Q1 저널 · 공개 데이터셋
- `docs/02_실행가이드.md` — 단계별 상세 가이드 + 트러블슈팅
- `docs/03_증류_프런티어모방.md` — 증류를 합법적으로 하는 법(폐쇄형 금지 이유 포함)

## 합법성 원칙
- 풀텍스트는 **오픈액세스만**(arXiv/OA). 유료 Q1 본문은 받지 않고 **메타+초록**만 색인.
- 증류 교사는 **오픈웨이트만**. 폐쇄형(Claude/GPT/Gemini) 출력 학습은 약관 위반 → 사용 안 함.
- 데이터셋은 각 라이선스 준수. 사용자 보유 PDF는 `data/my_papers/`에 두면 본인 책임 하에 색인.
