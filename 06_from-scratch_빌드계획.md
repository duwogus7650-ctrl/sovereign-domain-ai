# from-scratch 소형 모델 빌드 계획 (선택한 길)

목표: 가중치 100% 자체(from-scratch)이되, 검증된 방법은 그대로 베껴 출발선을 앞당긴 모터·로봇 도메인 특화 소형 모델. 좋은 소식 — '베낄 방법'이 지금은 거의 공개 매뉴얼 수준.

## 구조는 베끼되 발명하지 마라
2026년 오픈 모델들이 사실상 하나로 수렴: **디코더 온리 트랜스포머 + RMSNorm(pre-norm) + RoPE + SwiGLU + Grouped-Query Attention(GQA) + bias 제거 + (소형은) tied embedding.** 격언: "구조에서 혁신하지 말고 데이터·학습 동역학에서 혁신하라." 이 템플릿을 그대로 쓰는 게 정답.

## 베낄 레시피
- 코드/레시피: HuggingFace **Smol Training Playbook** + **SmolLM3(3B)** = 사실상 완전 공개된 교과서. 학습 프레임워크 **nanotron**.
- 완전 재현 레퍼런스: Allen AI **OLMo**(가중치·데이터·중간 체크포인트까지 공개).
- 처음 배울 땐: **카파시의 nanoGPT**.
- 토크나이저: 한국어가 Llama 토크나이저보다 15% 이상 잘 압축되는 게 아니면 그냥 Llama·Qwen 토크나이저 재사용(검증된 어휘 물려받기).
- 데이터: FineWeb-Edu(고품질 웹) + The Stack v2(코드) + SlimPajama + 한국어 + 도메인 데이터. `datatrove`로 중복제거.
- 컴퓨팅: 작은 실험은 H100 1장, 본격 솔로 작업은 **8×H100 단일 노드**가 스위트스폿.

## 솔직한 부분
- 코드·구조는 이제 '쉬운' 쪽(다 공개됨).
- **진짜 일·병목은 데이터, 그중에서도 한국어.** 전 세계 코퍼스의 1% 미만. AI Hub(aihub.or.kr)가 한국어 LLM 사전학습·instruction 데이터를 공개하지만 양·품질이 영어에 한참 못 미침. from-scratch 한국어 모델의 성패는 여기서 갈림.
- 능력 천장: 개인 컴퓨팅·데이터로 만든 소형 모델은 '쓸만하고 내 것이고 순수하지만' 프런티어는 아님. 기대치를 처음부터 맞출 것.

## 증명(아날로그)
**MiniLingua** — 13개 유럽 언어용 1B 모델을 from-scratch로 학습하고 코드·데이터까지 공개(대학팀, '영어 편중·프라이버시' 문제의식). '한국어+모터 도메인 소형 소버린'이 정확히 같은 모양. 충분히 가능.

## 단계별 계획
- **Stage 0 — 미니 모델로 파이프라인 학습**: nanoGPT로 ~100–300M을 단일 GPU에서 끝까지. 토크나이저→데이터→학습→평가 전 과정을 손에 익힘. (핵심 조언: 바로 0.5B에 돈 쓰지 말 것)
- **Stage 1 — from-scratch 본 학습**: SmolLM3 레시피·nanotron으로 0.5–1B, 데이터 믹스 + 8×H100.
- **Stage 2 — 사용 가능화·배포**: SFT(도메인+지시) → GGUF Q4 → Ollama(오프라인).

## 다음 한 수(미정)
- (A) Llama 스타일 구조(RoPE·RMSNorm·SwiGLU·GQA)를 박은 최소 nanoGPT급 학습 스크립트 스캐폴딩
- (B) 데이터 파이프라인(수집→정제→중복제거→토큰화) 설계

## 주요 출처
- codersera 'Self-Training a Small LLM (2026)', HuggingFace Smol Training Playbook, OLMo(Allen AI), TinyLlama, MiniLingua(arXiv 2512.13298), AI Hub
- 모델 지형: HF blog open-source-llms, techsy, mljourney, benchlm / 파인튜닝: effloow LoRA·QLoRA 가이드

> 다이어그램: `다이어그램/5_도메인_파인튜닝_파이프라인.svg`(파인튜닝 경로 참고), `7_from-scratch_단계별_계획.svg`
