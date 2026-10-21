# 💰 Financial RAG Agent

> 금융 문서 PDF 23개를 검색해서 **근거(파일명·페이지)와 함께 답하는 RAG Agent** 팀 프로젝트

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?logo=googlegemini&logoColor=white)
![Chroma](https://img.shields.io/badge/Chroma-FF6446)
![RAGAS](https://img.shields.io/badge/RAGAS-4B32C3)

| 항목 | 내용 |
|:---|:---|
| 기간 | 2026.10.20 ~ 2026.10.22 (3일) |
| 형태 | RAG Agent 실습 과제, 3인 실습 |
| 진행 방식 | 설계 방향(파이프라인, 추출·청킹·검색 기준)은 3명이 함께 논의해 결정 |
| AI 도구 | Claude(Claude Code)만 사용 |

과제 설명 원문은 [docs/assignment.md](docs/assignment.md)에 있어요.

<br>

## 📍 진행 상황

| 단계 | 상태 | 내용 |
|:---|:---:|:---|
| 1. 문서 추출 | ✅ 완료 | PDF → 페이지별 본문 + 마크다운 표 + 그래프·그림·수식 해석 |
| 2. 청킹 · 임베딩 | ✅ 완료 | 구조 반영 청킹 300/50, 500/60 → `gemini-embedding-2` → Chroma |
| 3. 검색 도구 | ⏳ 다음 | **300/50 + Hybrid(5:5) 확정, 리랭킹 없이 상위 5개를 답변 LLM에 전달** → 청크 + ID·출처 반환 |
| 4. Agent 설계 | ⬜ | 판단·행동 흐름, 종료 조건 |
| 5. 평가 루프 | ⏳ 진행 | 검색 평가 완료 (골든셋 4종 437문항, 청킹 2 × 검색 4), 답변 평가 + 개선 (golden_set 20문항 correctness 0.725 → 0.950, 처음 보는 30문항 0.850 → 0.883) |

<br>

## 📂 폴더 구조

```
financial-rag-agent/
├── 01_data/                 # 원문 PDF 23개 (강의 제공)
│   ├── financial_glossary_readable.pdf
│   ├── financial_life_guide.pdf
│   ├── standard_terms/      # 표준약관 15개
│   └── additional/          # 주택연금, 퇴직연금, 정책금융 등 6개
├── 02_questions/            # 질문 180문항 (강의 제공)
├── 03_answers/              # 기준 답안 (강의 제공)
├── 04_extracted/            # 추출 결과
│   ├── pages.jsonl          #   쪽마다 한 줄 ← 다음 단계(청킹)의 입력
│   ├── index.md             #   문서별 요약 + 비교 문서 링크
│   ├── review/              #   문서별 비교 md + 쪽 캡처
│   ├── image_check.md       #   숫자 검증을 못 통과한 그림 해석 (사람 검토용)
│   ├── gemini_tables.json   #   Gemini로 다시 뽑은 표 + 검증 결과
│   ├── layout.json          #   rapid-layout 영역 분석 결과 (490쪽)
│   └── image_regions.json   #   그림 영역 해석 + 검증 결과
├── 05_golden_set/           # 검색 평가용 골든셋 (아래 "검색 평가" 참고)
├── 05_make_golden_set.py    # 03_answers → golden_set(20문항 표본) + golden_set_provided(180문항 전체)
├── extraction/              # 추출 스크립트 (번호 = 실행 순서)
│   ├── 01_extract.py        #   ① PDF → 본문·표·그림 해석 합치기 → pages.jsonl
│   ├── 02_make_review_md.py #   ② 원문 캡처와 추출 결과 비교 문서
│   ├── 03_gemini_tables.py  #   ③ 깨진 표만 Gemini로 다시 뽑기 (숫자 검증)
│   └── 04_image_regions.py  #   ④ 그래프·그림·다이어그램·수식 (rapid-layout → Gemini/OCR)
├── 06_chunks/               # 청크 결과 (설정별 jsonl)
│   ├── chunks_300_50.jsonl  #   3,013개
│   └── chunks_500_60.jsonl  #   1,908개
├── indexing/                # 청킹·임베딩 스크립트 (번호 = 실행 순서)
│   ├── 01_chunk.py          #   ① pages.jsonl → 구조 반영 청크
│   ├── 02_embed.py          #   ② 청크 → gemini-embedding-2 → Chroma
│   ├── 03_search_test.py    #   ③ 두 설정에 같은 질문 던져 비교
│   └── common.py            #   벡터 DB 위치·임베딩 모델 설정
├── evaluation/
│   ├── 01_retrieval_eval.py #   골든셋 검색 평가 (Similarity·MMR·BM25·Hybrid)
│   └── 02_answer_eval.py    #   골든셋 답변 평가 (검색 → Gemini 답변 → Gemini 채점)
├── 07_eval/                 # 평가 결과 (아래 "평가 결과 관리" 참고)
│   ├── runs.jsonl           #   실행 이력: 실행마다 요약 한 줄씩 쌓임
│   └── {골든셋 이름}/       #   최신 실행 결과: 설정별 표 txt + 질문별 jsonl
├── chroma_db/               # 벡터 DB (커밋 안 함, 02_embed.py로 생성 — 한글 경로면 다른 곳, 아래 참고)
├── docs/
│   ├── assignment.md        # 과제 설명
│   ├── chunking_todo.md     # 청킹 단계에서 해결할 것
│   ├── 비용기록.md          # 유료 API 호출마다 토큰·비용 한 줄 (평가 스크립트가 자동 추가)
│   └── 비용추정.md          # 추출 단계 Gemini 비용 (약 $1.8)
└── 회고.md                  # 단계마다 만난 문제와 해결 과정
```

<br>

## 🚀 실행 방법

```bash
pip install pdfplumber python-dotenv langchain-google-genai
python extraction/01_extract.py          # 04_extracted/pages.jsonl 생성 (약 2분)
python extraction/02_make_review_md.py   # 04_extracted/index.md, review/*.md 생성
```

> `01_extract.py`를 고쳤다면 `02_make_review_md.py`도 다시 돌려야 비교 문서에 반영돼요.

**Gemini로 표 다시 뽑기 (선택)**

pdfplumber로 안 되는 표는 `03_gemini_tables.py`의 `TARGETS`에 문서·쪽·영역을 적고 돌려요.
결과(`04_extracted/gemini_tables.json`)는 이미 레포에 있어서, **새 표를 추가할 때만** 필요해요.

```bash
# .env 에 GEMINI_API_KEY=... 필요 (.env는 커밋하지 않음)
python extraction/03_gemini_tables.py    # 표 영역 캡처 → Gemini → 숫자 검증
python extraction/01_extract.py          # 검증 통과한 표를 pdfplumber 표 대신 사용
python extraction/02_make_review_md.py
```

**그래프·그림 뽑기 (선택)**

결과(`layout.json`, `image_regions.json`)가 레포에 있어서 평소엔 `01_extract.py`만 돌리면 돼요. 새로 돌릴 때만:

```bash
pip install rapid-layout rapidocr onnxruntime
python extraction/04_image_regions.py    # 레이아웃 분석(약 15분, 한 번만) → 그림 처리(Gemini 6개 동시)
python extraction/01_extract.py
python extraction/02_make_review_md.py   # image_check.md(검토 목록)도 같이 생성
```

> OpenCV는 Windows에서 **한글 경로에 파일을 저장하지 못해요.** 시각화 저장이 안 되면 경로를 확인하세요.

<br>

## 🧱 청킹 · 임베딩

```bash
pip install langchain-text-splitters langchain-chroma
python indexing/01_chunk.py        # 06_chunks/chunks_300_50.jsonl, chunks_500_60.jsonl (1분 이내)
python indexing/02_embed.py        # chroma_db/ 생성 (약 5~10분, 이미 넣은 청크는 건너뜀)
python indexing/03_search_test.py  # 두 설정 검색 비교 (질문을 인자로 넣을 수도 있음)
```

> ⚠️ **Windows에서 프로젝트 경로에 한글이 있으면** Chroma가 색인 파일(`.bin`)을 저장하지 못해서 다시 열 때 `Error loading hnsw index`가 나요.
> 그래서 경로에 한글이 있으면 자동으로 `C:/Users/Public/financial-rag-agent/chroma_db`에 저장해요(실행 시 알림).
> 다른 곳에 두려면 `.env`에 `CHROMA_DIR=영문경로`를 넣으세요.
>
> ⚠️ Chroma는 HNSW 색인을 1,000개 단위로만 디스크에 써서, 나머지가 로그에만 남으면 **같은 질문의 검색 결과가 바뀌었어요**([회고](회고.md) 15번).
> `02_embed.py`는 컬렉션을 만들 때 청크 수만큼 모았다가 한 번에 저장하게 설정해요. 중간에 끊겼다면 `chroma_db`를 지우고 처음부터 돌리세요.

**비교 실험 설정** (나머지는 모두 같음)

| 설정 | 크기 / 겹침 (글자) | 청크 수 | 길이 중앙값 |
|:---|:---:|---:|---:|
| `chunks_300_50` | 300 / 50 | 3,013 | 270 |
| `chunks_500_60` | 500 / 60 | 1,908 | 438 |

**공통 전처리 (구조 반영 분할)**

| 대상 | 처리 |
|:---|:---|
| 머리글·꼬리말 | 쪽 번호, 반복 제목, 한국은행 경고문 등 쪽 위·아래 반복 줄 삭제 |
| 표 | 바로 위 제목·단위 줄(캡션 패턴 또는 문장이 아닌 40자 미만 줄, 최대 4줄 + 짧은 제목 덩어리) + 바로 아래 각주(주·자료·출처·*·※)와 묶음. 크기를 넘으면 **행 단위로 나누고 조각마다 제목·헤더 반복** |
| 그래프·그림·수식 | 쪼개지 않고 통째로 한 청크 (설명 문장 포함) |
| 본문 | `RecursiveCharacterTextSplitter`, **쪽 단위**로 나눔 (출처 쪽 번호 정확) |

**청크 메타데이터:** `chunk_id`, `source`, `page`, `doc_group`(`glossary` / `standard_terms` / `life_guide` / `additional`), `block_type`(`text` / `table` / `figure`)

> 기본 negative 세트는 **용어집 + 약관만 검색**하는 조건이라 `doc_group`으로 필터해서 써요.

<br>

## 📊 검색 평가 (골든셋)

```bash
pip install kiwipiepy rank-bm25 langchain-classic langchain-community
python evaluation/01_retrieval_eval.py                    # 05_golden_set/*.jsonl 전부 (파일마다 따로)
python evaluation/01_retrieval_eval.py 05_golden_set/xxx.jsonl
```

> `kiwipiepy`가 없다는 오류가 나면 패키지를 설치한 가상환경의 파이썬으로 실행하세요 (예: `../../.venv/Scripts/python.exe evaluation/01_retrieval_eval.py`).

- 결과: `07_eval/{골든셋 이름}/retrieval_{청킹}.txt`(난이도별), `..._by_type.txt`(근거 유형별), `retrieval_rows.jsonl`(질문별), `07_eval/runs.jsonl`(실행 이력)
- 정답: 문서 + 쪽 (약관 `kofia-terms-…-{버전}-{조}`는 버전 → 약관 파일, 조 → 쪽 / 용어집은 파일 단위 / 정답 문서 없는 문항은 제외)
- 검색 설정은 강의 스크립트와 같음: k=5, fetch_k=20, MMR λ=0.5, Hybrid 0.5:0.5, BM25는 Kiwi 토큰화
- latency는 질문 임베딩을 뺀 **순수 검색 시간** (질문 임베딩은 한 번만 만들어 모든 방식이 공유)

| 골든셋 | 문항 (평가) | 근거 유형 | 만든 방법 |
|:---|---:|:---|:---|
| `golden_set_table_image` | 189 | table 109 · image 58 · formula 18 · paragraph 2 · none 2 | 직접 작성 (표·그림·수식) |
| `golden_set_paragraph` | 100 | paragraph 98 · none 2 | 직접 작성 (본문 문단, 23개 문서 전체) |
| `golden_set_provided` | 180 (148) | paragraph 135 · table 11 · image 2 · none 32 | 강의 제공 03_answers 전체를 변환 |
| `golden_set` | 20 (16) | paragraph 11 · table 3 · image 2 · none 4 | 제공 문항 중 20개 표본 (`provided`에 포함) |
| `golden_set_all` | 469 (437) | paragraph 235 · table 120 · image 60 · formula 18 · none 4 | table_image + paragraph + provided 합침 |

- 괄호 = 정답 문서가 있어서 검색 평가에 들어간 문항 수
- `provided`의 근거 유형은 20문항만 직접 붙였고 나머지는 **자동 추정**(답의 숫자가 본문엔 없고 표/그림에만 있으면 table/image)
- `golden_set_all`은 파일로 두지 않고 세 파일을 이어 붙여 평가했어요 (`05_golden_set/`에 두면 기본 실행 때 중복 평가됨)

**`golden_set_all` 437문항**

| 청킹 | 검색 | hit@1 | hit@5 | recall@5 | MRR |
|:---|:---|---:|---:|---:|---:|
| 300/50 | Similarity | 0.675 | 0.906 | 0.850 | 0.767 |
| 300/50 | MMR | 0.675 | 0.890 | 0.798 | 0.759 |
| 300/50 | BM25 | 0.597 | 0.822 | 0.754 | 0.686 |
| **300/50** | **Hybrid** | 0.703 | **0.938** | **0.877** | 0.795 |
| 500/60 | Similarity | 0.638 | 0.892 | 0.837 | 0.737 |
| 500/60 | MMR | 0.638 | 0.854 | 0.769 | 0.729 |
| 500/60 | BM25 | 0.586 | 0.858 | 0.787 | 0.694 |
| 500/60 | Hybrid | **0.719** | 0.920 | 0.861 | **0.801** |

**근거 유형별 Hybrid (hit@5 / MRR)**

| 유형 | 문항 | 300/50 | 500/60 |
|:---|---:|:---|:---|
| paragraph | 235 | **0.919 / 0.772** | 0.877 / 0.755 |
| table | 120 | 0.967 / 0.802 | **0.975 / 0.826** |
| image | 60 | 0.967 / 0.871 | **0.983 / 0.917** |
| formula | 18 | 1.000 / 0.847 | 1.000 / **0.926** |

> 표 제목 붙이기(아래 답변 평가 개선) 뒤 다시 돌린 값이에요. 바뀌기 전과 모두 ±0.02 이내이고, 437문항 중 2문항이 새로 맞고 3문항이 빠졌어요 (청크 순위가 조금씩 바뀜).

**결정: 300/50 + Hybrid**
- 검색 방식은 모든 유형에서 Hybrid가 가장 좋음
- 청킹은 유형마다 갈림: paragraph는 300/50, table·image·formula는 500/60이 조금 나음 (hit@1·MRR 전체도 500/60이 약간 높음)
- 그래도 300/50을 고른 이유: 문항이 가장 많은 paragraph에서 차이가 가장 큼(hit@5 10문항 차이), table에서 500/60이 더 맞힌 건 120문항 중 1문항. 정답을 5개 안에 넣는 것(hit@5·recall@5)이 답변 생성에 더 중요
- 청크 종류별로 섞는 방법(본문 300/50 + 표 500/60)도 검토했지만 얻을 수 있는 게 1~2문항이라 하지 않음. 그림·수식 청크는 두 설정에서 똑같음
- 남은 약점은 hit@1(약 0.70) → 다음 단계에서 리랭커 또는 상위 5개를 모두 LLM에 넘기는 방식으로 보완

**hit@1을 올리려는 추가 실험 (300/50 Hybrid 기준)**

| 실험 | 결과 | 판단 |
|:---|:---|:---|
| Hybrid 가중치 (벡터:BM25 0:10 ~ 10:0) | provided로 고른 6:4가 확인용 셋에서 5:5 대비 ±1~2문항 | **5:5 유지.** 섞는 게 중요하고 0.5~0.7 사이는 평평함 |
| 리랭커 `bge-reranker-v2-m3` (Hybrid 상위 10개 → 상위 5개) | paragraph 무작위 30문항: hit@1 0.633 → **0.767**, MRR 0.756 → 0.836, hit@5 그대로 0.967 | 효과는 있지만 **CPU에서 질문당 약 15초** → 지금은 안 씀 |

**결정: 리랭킹 없이 Hybrid 상위 5개를 모두 답변 LLM에 넘김**
- hit@5가 이미 0.94~0.97이라, 정답 근거는 대부분 5개 안에 들어 있음 → 맞는 근거 고르기는 답변 생성 LLM에게 맡김 (추가 비용 없음)
- 답변 평가에서 근거를 잘못 고르는 일이 많으면 그때 리랭킹 추가: Gemini로 후보 순위 매기기(질문당 LLM 1회) 또는 API 리랭커
- 리랭커 실험은 30문항 표본이라 4문항 차이도 흔들릴 수 있음. 실험 코드는 레포에 넣지 않음 (자세한 과정은 [회고](회고.md) 16번)

> 처음 기록한 500/60 수치(table_image Similarity hit@5 0.810 등)는 Chroma 색인 문제로 틀렸어요. 색인을 다시 만든 뒤 모든 골든셋을 다시 돌린 값이 위 표예요.
> 골든셋별·난이도별·근거 유형별 표는 `07_eval/{골든셋 이름}/`에 있어요.

**평가 결과 관리**

결과는 DB 없이 JSONL 파일로 남겨요. 줄마다 JSON 하나라서 git diff로 바뀐 점이 보이고, pandas로 바로 읽을 수 있어요.

| 파일 | 내용 | 저장 방식 | 용도 |
|:---|:---|:---|:---|
| `07_eval/runs.jsonl` | 실행 한 번 = 한 줄: 실행 시각, 골든셋, 문항 수, 조건(임베딩 모델·k·fetch_k·MMR λ·Hybrid 가중치), 청킹×검색별 hit@1·hit@5·recall@5·MRR·latency | **이어 붙임** (지워지지 않음) | 실행끼리 비교, 결정의 근거 |
| `07_eval/{골든셋}/retrieval_rows.jsonl` | 질문 × 청킹 × 검색마다 한 줄: 점수 + 가져온 청크(파일·쪽) | 실행마다 **덮어씀** | 틀린 문항 찾기 |
| `07_eval/{골든셋}/retrieval_*.txt` | 난이도별·근거 유형별 표 | 실행마다 **덮어씀** | 사람이 읽는 요약 |

```json
{"run_at": "2026-10-21T15:30:00", "golden_set": "golden_set", "count": 16,
 "params": {"embedding": "gemini-embedding-2", "k": 5, "fetch_k": 20, "mmr_lambda": 0.5, "hybrid_weights": [0.5, 0.5]},
 "results": {"chunks_300_50/Hybrid": {"hit@1": 0.562, "hit@5": 0.938, "recall@5": 0.844, "mrr": 0.74, "latency_ms": 17.0}, "...": {}}}
```

```python
import pandas as pd

# 실행 이력: 골든셋별로 300/50 Hybrid hit@5가 어떻게 바뀌었는지
runs = pd.read_json("07_eval/runs.jsonl", lines=True)
runs["hybrid_hit5"] = runs["results"].map(lambda r: r["chunks_300_50/Hybrid"]["hit@5"])
print(runs[["run_at", "golden_set", "hybrid_hit5"]])

# 질문별: Hybrid가 5개 안에 못 맞힌 문항
rows = pd.read_json("07_eval/golden_set_all/retrieval_rows.jsonl", lines=True)
print(rows[(rows.config == "chunks_300_50") & (rows.strategy == "Hybrid") & (rows["hit@5"] == 0)][["case_id", "retrieved"]])
```

- 조건(청킹·가중치·리랭커 등)을 바꿔 실험할 때는 `runs.jsonl`에 쌓인 줄끼리 비교해요. 질문별 결과까지 남기려면 돌리기 전에 `07_eval/{골든셋}/` 폴더를 복사해 두세요.
- `runs.jsonl`은 이 기능을 추가한(2026-10-21) 뒤의 실행부터 기록돼요. 그전 결과는 위 표에 정리돼 있어요.

<br>

## 📝 답변 평가 (골든셋)

```bash
python evaluation/02_answer_eval.py                          # 05_golden_set/golden_set.jsonl (20문항)
python evaluation/02_answer_eval.py 05_golden_set/xxx.jsonl
python evaluation/02_answer_eval.py --neighbors=0             # 이웃 청크 없이 상위 5개만 (비교용)
python evaluation/02_answer_eval.py --prompt=base             # 프롬프트 규칙 묶음 (base / A / AB / ABC, 기본 ABC)
python evaluation/02_answer_eval.py 05_golden_set/golden_set_provided.jsonl --sample=30
                                                              # 무작위 30문항 확인 (튜닝용 golden_set 문항은 빼고, seed 0)
```

검색은 확정한 기준 설정(300/50 + Hybrid 5:5, 상위 5개)에 **이웃 청크 ±1**(아래 개선 실험)을 붙여서 넘기고, Gemini(`gemini-3.6-flash`)가 근거만으로 답한 뒤(프롬프트 ABC, 아래 프롬프트 개선) 같은 모델이 채점해요.
기본 positive(POS001~060)·negative(NEG) 문항은 과제 조건대로 **용어집 + 표준약관만 검색**해요.

**채점 방식:** RAGAS 대신 문항마다 채점 LLM 1회
- 골든셋에 기준 답안·`must_include`·`must_not_include`·`expected_behavior`가 이미 있어서 그대로 기준으로 씀
- RAGAS는 지표마다 LLM을 여러 번 불러서 비용이 더 들고, 토큰 사용량을 따로 모으기 어려움
- 근거 충실성은 RAGAS Faithfulness와 같은 방식 (답을 주장으로 쪼개고 → 근거에 있는 비율)

| 지표 | 뜻 | 계산 |
|:---|:---|:---|
| correctness | 기준 답안과 맞나 | LLM 채점 (1 / 0.5 / 0) |
| faithfulness | 지어낸 내용이 없나 | 근거로 뒷받침되는 주장 비율 |
| must_include | 꼭 들어가야 할 내용 | 충족한 항목 비율 |
| must_not | 피해야 할 오류 | 하나라도 했으면 1 (낮을수록 좋음) |
| behavior_ok | 기대 행동 (근거 인용·거절·추가 정보 요청 등) | LLM 채점 |
| citation_ok | 답에 적은 `[파일 p쪽]` 중 정답 쪽이 있나 | 규칙 (정답 문서 없는 문항 제외) |
| outcome | 성공 / **검색 실패** / **답변 실패** | 성공 = correctness ≥ 0.5 + must_not 없음 + behavior_ok. 실패면 LLM에 넘긴 근거에 정답 쪽이 있었는지로 원인 구분 |

결과: `07_eval/{골든셋}/answer_rows.jsonl`(문항별 답·근거·점수·채점 이유), `answer_summary.txt`, `07_eval/runs.jsonl`(`eval: "answer"`), [docs/비용기록.md](docs/비용기록.md)

**1차 결과 (`golden_set` 20문항, run_id `answer-golden_set-20261021-093244`)**

| 세트 | 문항 | correctness | faithfulness | must_include | must_not | behavior_ok | citation_ok | 성공 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 전체 | 20 | 0.725 | 0.975 | 0.746 | 0 | 0.95 | 1.00 | 17 |
| positive | 14 | 0.714 | 1.000 | 0.744 | 0 | 1.00 | 1.00 | 12 |
| negative | 6 | 0.750 | 0.917 | 0.750 | 0 | 0.83 | 1.00 | 5 |

- 지어낸 내용은 거의 없음 (faithfulness 0.975, 근거에 없는 주장 1개: NEG001에서 확인처를 안내한 문장)
- 답변 시간 평균 약 8.8초 (검색은 0.05초)
- 비용: 답변 생성 $0.14 + 채점 $0.13 = **약 $0.27** (출력 토큰의 대부분이 생각 토큰)

**실패 3문항 분석**

| 문항 | 판정 | 실제 원인 |
|:---|:---|:---|
| POS080 (현금흐름표 계산) | 답변 실패 | **검색 문제.** p11의 설명 청크는 찾았지만 숫자가 든 표 청크(p11:6~8)는 못 찾음. 표 조각에 "현금흐름표" 제목이 안 붙어 있음 |
| ADD-SE06 (34.2% 불일치) | 답변 실패 | **검색 문제.** p2는 찾았지만 DTA 계산 방식 설명이 있는 p3를 못 찾음 |
| NEG017 (전산장애 책임) | 답변 실패 | 답변 문제. 책임이 확정되지 않는다는 건 맞았지만 "주문 기록·적용 약관 확인" 안내가 빠짐 |

> 💡 **쪽 단위 hit@5는 너그러워요.** 정답 쪽의 청크가 하나라도 오면 hit로 세서, 정작 필요한 청크(표 숫자, 다음 쪽 설명)가 빠져도 "답변 실패"로 분류돼요.
> 두 문항 모두 LLM은 지어내지 않고 "찾을 수 없다"고 답해서 근거 충실성은 지켰어요.
> 다음 개선 후보: 쪼개진 표 조각에 표 제목 붙이기, 근거가 이어지는 다음 쪽 청크 같이 넘기기.

**개선 실험: 근거가 빠지는 문제 (`golden_set` 20문항, 한 번에 하나씩 바꿈)**

| 실험 | 바꾼 것 | 넘긴 청크 | correctness | 완전 정답(1.0) | faithfulness | must_include | 성공 | 비용 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| A. 기준 | - | 5 | 0.725 | 11 | 0.975 | 0.746 | 17 | $0.27 |
| B. 표 제목 | `01_chunk.py`: 표 위 짧은 제목 줄도 표에 붙임 | 5 | 0.775 | 12 | 0.958 | 0.796 | 17 | $0.25 |
| **C. 표 제목 + 이웃 청크** | 검색된 청크마다 앞뒤 1개 + 걸친 표는 모든 조각 | 평균 15 | **0.825** | **14** | **0.983** | **0.838** | 17 | $0.31 |

run_id: A `answer-golden_set-20261021-093244`, B `…-094748`, C `…-095141` (`07_eval/runs.jsonl`, `chunks_md5`로 청크 버전 구분)

- **표 제목 붙이기:** 제목 없던 표 조각이 많았음 (300/50 청크 299개의 글이 바뀜, 그중 표 136개, 나머지는 제목 줄이 빠진 본문). 예: 현금흐름표 조각들은 "현금흐름표(예시)" 줄이 위 본문 청크에만 있었음
  - 줄 끝이 `다.`·`요.`·`?`·`:`로 끝나거나 앞 표의 `자료:` 각주면 제목으로 안 봄
  - 다시 임베딩은 글이 바뀐 청크만 ($0.015, `02_embed.py`가 같은 글은 옛 벡터를 옮김)
  - 검색 지표는 거의 그대로 (위 검색 평가 표), 답할 때 표 숫자가 무엇인지 LLM이 알 수 있게 됨
- **이웃 청크:** POS080(현금흐름표 계산)은 C에서만 맞음. 설명 청크(p11:5)만 검색돼도 표 조각 3개(p11:6~8)가 같이 넘어가서 계산까지 정확히 함
  - ±1만으로는 첫 표 조각만 와서, **범위 끝이 표 조각이면 그 표의 나머지 조각까지** 넓힘 (첫 줄이 같은 연속 표 청크)
  - 넘기는 글이 평균 약 3,700자(최대 9,400자)로 늘어서 입력 토큰이 2.5배, 비용은 20문항당 +$0.04 정도
- **결정: C를 기본값으로** (`--neighbors` 기본 1)

**아직 남은 실패 / 주의할 점**
- ADD-SE06: 필요한 근거(p3의 DTA 계산 방식)가 검색된 p2에서 여러 청크 떨어져 있어서 ±1로도 안 옴. B에서 0.5로 맞은 건 우연에 가까움
- NEG005(2027년 세액공제): A는 성공, B·C는 실패. 모델이 문서 기준일(2026-09-03)을 모르니 "기준일 명시"를 못 함 → 아래 프롬프트 개선 B
- NEG017: 세 번 모두 "주문 기록 확인" 안내가 빠짐 → 아래 프롬프트 개선 A
- 같은 설정으로 다시 돌려도 1~2문항은 결과가 바뀜 (ADD-PF01, NEG009가 0.5 ↔ 1.0). 20문항에서 1~2문항 차이는 잡음으로 보고, **원인이 분명한 변화(POS080)만 근거로 판단**함

**프롬프트 개선 (이웃 청크 ±1 고정, 규칙 묶음을 하나씩 더함)**

점수가 깎인 문항의 패턴: ① 마지막 "주의할 점·추가로 확인할 것"을 빼먹음 ② "문서에서 찾을 수 없다"를 너무 빨리 말함 ③ 문서 기준일을 모름.
기준 답안은 거의 다 "결론 → 근거 → 주의점" 구조라서 그 순서를 프롬프트에 넣었어요. 규칙 원문은 `02_answer_eval.py`의 `EXTRA_RULES`에 있어요.

| 규칙 | 내용 |
|:---|:---|
| A. 답변 구조 | 결론 → 근거(계산 과정 포함) → 주의할 점 (예시 수치는 예시라고, 제도마다 조건이 다르면 그렇다고, 확정 못 하는 판단은 확인할 자료 안내) |
| B. 문서 기준일 | 2026-09-03 (강의 답안에만 나오는 corpus snapshot 날짜). **미래·최신을 물을 때만** 기준일을 밝히고 최신 자료 안내 |
| C. 부족한 근거 | 있는 만큼은 답하기, 계산은 근거 숫자로 직접, 숫자 불일치는 오타 단정 대신 계산 방식 차이 설명, 투자 권유 거절 시에도 위험·분산 설명 |

*튜닝셋 (`golden_set` 20문항, 이 문항들을 보고 고침)*

| 프롬프트 | correctness | 완전 정답 | faithfulness | 성공 |
|:---|---:|---:|---:|---:|
| base | 0.825 | 14 | 0.983 | 17 |
| A | 0.925 | 17 | 1.000 | 19 |
| AB | 0.950 | 18 | 0.951 | 17 |
| ABC | 0.975 | 19 | 0.983 | 19 |
| **ABC (B 수정)** | 0.950 | 18 | 0.993 | **20** |

- AB·ABC 처음 버전은 상관없는 답에도 "2026-09-03 기준"을 붙여서 채점에서 "근거에 없는 숫자"로 감점(must_not 3문항) → B를 "미래·최신 질문에만"으로 좁힘

*처음 보는 30문항 (`golden_set_provided`에서 튜닝 20문항 빼고 무작위, positive 24 · negative 6)*

| 프롬프트 | correctness | 완전 정답 | faithfulness | must_not | 성공 | 답변 시간 |
|:---|---:|---:|---:|---:|---:|---:|
| base | 0.850 | 22 | **0.997** | **0** | **27** | 8.4초 |
| **ABC** | **0.883** | **24** | 0.974 | 1 | 26 | 11.0초 |

- 튜닝셋에서는 +0.125, 처음 보는 문항에서는 **+0.033** → 차이만큼은 튜닝셋에 맞춘 효과
- 좋아짐 4, 나빠짐 2. 네거티브 6문항 correctness 0.833 → **1.000**, 포지티브는 0.854로 같음
- 새 부작용: C의 분산 규칙 때문에 **상관없는 약관 질문에도 분산투자 문장이 붙음** (POS057·POS059, faithfulness 하락의 원인)
- 두 프롬프트 다 못 하는 것: 정보가 부족할 때 **되묻기** (NEG014·NEG016, `ask_clarifying_question`)
- 답이 길어져서 답변 시간 +2.6초, 비용 +25%
- **결정: ABC를 기본값으로** (`--prompt` 기본 ABC). 다음 후보는 분산 문장을 투자 권유 요청 때만 쓰게 하기, 되묻기 규칙. 이 30문항을 보고 고치게 되니까 확인은 **다른 seed의 30문항**으로
- 프롬프트 실험 비용 약 $2.5 (20문항 4회 + 30문항 2회)

<br>

## 📄 추출 결과 형식

`04_extracted/pages.jsonl` — PDF 한 쪽이 한 줄이에요.

```json
{
  "source": "additional/policy_finance_debt_adjustment_2026.pdf",
  "page": 12,
  "text": "표를 뺀 본문",
  "tables": ["| 구분 | 2021 | ... |\n| --- | --- | ... |\n..."],
  "table_sources": ["pdfplumber"],
  "content": "본문과 표를 위→아래 읽는 순서대로 합친 것"
}
```

| 필드 | 설명 | 용도 |
|:---|:---|:---|
| `source` | `01_data/` 기준 경로 | 답변의 출처 표시 |
| `page` | 1부터 시작하는 쪽 번호 | 답변의 출처 표시 |
| `text` | 표 영역을 뺀 본문 | 본문만 필요할 때 |
| `tables` | 그 쪽의 마크다운 표 목록 | 표만 따로 다룰 때 |
| `table_sources` | 표마다 `pdfplumber` 또는 `gemini` | 평가 때 Gemini 표가 원인인지 구분 |
| `figures` | 그래프·그림 해석 (`type`, `markdown`, `from`) | 그림에서 온 정보만 따로 다룰 때 |
| `content` | 본문 + 표 + 그림 해석 (읽는 순서, `[그래프]`·`[그림]`·`[수식]` 표시, 그림마다 `설명:` 문장) | **청킹에는 이걸 사용** |

**현재 수치:** 23개 문서 · 490쪽 · 표 168개(그중 Gemini 3개) · 그래프·그림·수식 130개(그래프 96, 그림·절차도 28, 수식 4, 이미지 속 글자 2) · 글자가 거의 없는 쪽(표지·간지) 24쪽

<br>

## 🔍 추출 방식 요약

pdfplumber의 기본 표 인식(`find_tables`)만으로는 실제 문서의 표가 많이 깨져서, 아래 처리를 차례로 더했어요.
자세한 과정은 [회고.md](회고.md)에 있어요.

| 문제 | 처리 |
|:---|:---|
| 표 글자가 본문에도 섞임 | 표 영역 안 글자는 본문에서 제외 |
| 목차·강조 박스가 표로 잡힘 | 1행·1열, 데이터 칸이 전부 빈 표, 1000자 넘는 셀은 표에서 제외 |
| 바깥 세로 테두리가 없는 표 | 표 양 끝에 세로선을 지정해 다시 인식 |
| 병합 셀이 빈칸(`None`)으로 나옴 | 그 칸을 덮는 원래 셀 값으로 채움 |
| 화살표 줄에서 표가 위아래로 쪼개짐 | 헤더 행의 열 경계를 이어받아 하나로 합침 |
| 점선 행 구분선을 인식 못 함 | 점들을 이어 가로선으로 만들어 넘김 |
| 들여쓰기 항목 아래 점선이 칸 중간에서 시작 | 점선을 가장 가까운 세로선까지 늘림 |
| 들여쓰기 계층(포함 관계)이 사라짐 | `가계신용(A+B) > 가계대출(A) > 주택담보대출` 경로로 표시 |
| 두 쪽을 펼친 페이지에서 좌우 문장이 섞임 | 왼쪽 반 / 오른쪽 반을 따로 추출 |
| 배경 상자 때문에 같은 표가 두 번 나옴 | 절반 넘게 겹치는 표는 중복으로 제거 |
| `(cid:49049)` 같은 깨진 기호 | `→`로 치환 |
| 위 규칙으로도 안 되는 표·수식 | 표 영역 캡처를 Gemini로 다시 뽑고, **숫자가 원문에 있는지 검증**한 것만 사용 |
| 그래프·그림 속 정보가 빠짐 | rapid-layout으로 영역을 찾아 그래프·그림은 Gemini, 이미지 속 글자는 RapidOCR로 뽑고, Gemini 숫자는 **OCR로 검증** |
| 장식·장 번호 배지 | Gemini가 "장식"으로 분류하거나 너무 짧은 결과는 제외 |
| 벡터로 그린 절차도가 뒤섞임 | 작은 글상자 3개 이상이 가로로 이어지면 다이어그램으로 보고 Gemini로 흐름 추출 |
| 수식 폰트·분수가 깨짐 (`max min× 채무과중도`) | 깨짐 문자·분수 막대·`=`로 끝나는 줄을 찾아 Gemini로 한 줄 수식으로 |
| 값이 안 적힌 선그래프 | 그림마다 **설명 문장**(추세·흐름)을 받아서 살림 |
| Wingdings 글머리표가 깨진 문자로 나옴 | `•`로 치환 |

<br>

## ✅ 결과 검토하는 법

1. [04_extracted/index.md](04_extracted/index.md)에서 문서를 고른다.
2. 비교 문서에서 **왼쪽 원문 캡처 ↔ 오른쪽 추출 결과**를 쪽마다 비교한다.
3. 이상한 곳은 쪽 번호와 함께 팀에 공유한다.
4. [04_extracted/image_check.md](04_extracted/image_check.md)의 그림 해석 11개를 원본과 비교해서, 맞으면 `image_regions.json`에서 `status`를 `"ok"`로 바꾼다.

> 아직 완벽하지 않은 표 목록은 [docs/chunking_todo.md](docs/chunking_todo.md)의 5번에 정리돼 있어요.
