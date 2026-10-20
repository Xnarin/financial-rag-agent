# 💰 Financial RAG Agent

> 금융 문서 PDF 23개를 검색해서 **근거(파일명·페이지)와 함께 답하는 RAG Agent** 팀 프로젝트

과제 설명 원문은 [docs/assignment.md](docs/assignment.md)에 있어요.

<br>

## 📍 진행 상황

| 단계 | 상태 | 내용 |
|:---|:---:|:---|
| 1. 문서 추출 | ✅ 완료 | PDF → 페이지별 본문 + 마크다운 표 |
| 2. 청킹 · 임베딩 | ⏳ 다음 | [docs/chunking_todo.md](docs/chunking_todo.md)에 할 일 정리 |
| 3. 검색 도구 | ⬜ | 상위 k개 청크 + ID·출처 반환 |
| 4. Agent 설계 | ⬜ | 판단·행동 흐름, 종료 조건 |
| 5. 평가 루프 | ⬜ | 지표, 골든셋, 비교 실험 |

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
├── extraction/              # 추출 스크립트 (번호 = 실행 순서)
│   ├── 01_extract.py        #   ① PDF → 본문·표·그림 해석 합치기 → pages.jsonl
│   ├── 02_make_review_md.py #   ② 원문 캡처와 추출 결과 비교 문서
│   ├── 03_gemini_tables.py  #   ③ 깨진 표만 Gemini로 다시 뽑기 (숫자 검증)
│   └── 04_image_regions.py  #   ④ 그래프·그림·다이어그램·수식 (rapid-layout → Gemini/OCR)
├── docs/
│   ├── assignment.md        # 과제 설명
│   ├── chunking_todo.md     # 청킹 단계에서 해결할 것
│   └── 비용추정.md          # 추출 단계 Gemini 비용 (약 $1.8)
└── 회고.md                  # 추출 단계에서 만난 문제와 해결 과정
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
