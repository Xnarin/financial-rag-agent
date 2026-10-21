# 03_answers의 180문항에서 20문항을 골라 평가용 골든셋을 만든다.
# 추가 PDF 6 + 생활 가이드 4 + 용어사전 2 + 표준약관 2 + 거절 문항 6 = 20
# 180문항 전체도 같은 형식으로 golden_set_provided.jsonl에 저장한다.
import json
import re

def load(name):
    with open(f"03_answers/{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

rows = {r["case_id"]: r for name in ["additional_cases", "positive_life_guide_cases", "positive_cases", "negative_cases"]
        for r in load(name)}

# 문항 → 답의 근거가 어디 있는지 (paragraph=본문, table=표, image=그래프·그림, none=근거 없음)
# 04_extracted/pages.jsonl에서 답의 숫자·문장이 본문/표/그림 중 어디에만 있는지 보고 직접 붙였다.
CONTEXT_TYPE = {
    # 추가 PDF: 파일마다 1개
    "ADD-HP01": "paragraph",
    "ADD-PF01": "paragraph",
    "ADD-HF03": "paragraph",
    "ADD-RP03": "paragraph",
    "ADD-IN05": "table",      # 해지환급금 공제 예시 표
    "ADD-SE06": "table",      # 자영업 가구 재무상황 표 (1.78억, 7.38억)
    # 생활 가이드
    "POS061": "paragraph",
    "POS067": "image",        # 막대그래프 (월 저축액)
    "POS068": "image",        # 원형 차트 (투자성향별 포트폴리오), 추출 결과에 수치 없음
    "POS080": "table",        # 현금흐름표 예시
    # 용어사전·표준약관
    "POS001": "paragraph",
    "POS041": "paragraph",
    "POS019": "paragraph",
    "POS044": "paragraph",
    # 거절·주의 문항: 카테고리별 1개
    "NEG001": "none",
    "NEG009": "none",
    "NEG021": "none",
    "NEG029": "paragraph",    # 약관 근거로 답해야 하는 프롬프트 공격 문항
    "NEG005": "none",
    "NEG017": "paragraph",    # 약관 근거로 잘못된 전제를 바로잡는 문항
}
picked = [rows[c] for c in CONTEXT_TYPE]

def file_name(r):
    if "source_file" in r:
        return r["source_file"]
    src = (r["expected_source_ids"] or [""])[0]
    if src.startswith("fss-life-guide"):
        return "financial_life_guide.pdf"
    if src.startswith("fsc-glossary"):
        return "financial_glossary_readable.pdf"
    return None  # ponytail: kofia-terms ID → 표준약관 파일 매핑표가 없음, 생기면 채우기

def source_group(r):
    # 자료 묶음별로 성능을 나눠 보기 위한 열
    if "source_file" in r:
        return "additional"
    src = (r["expected_source_ids"] or [""])[0]
    if src.startswith("kofia-terms"):
        return "standard_terms"
    if src.startswith("fsc-glossary"):
        return "glossary"
    if src.startswith("fss-life-guide"):
        return "life_guide"
    return "none"  # 근거 문서가 없는 거절 문항

def to_golden(r, context_type):
    return {
    "case_id": r["case_id"],
    "source_group": source_group(r),
    "context_type": context_type,
    "set": r["set"],
    "category": r["category"],
    "difficulty": r["difficulty"],
    "question": r["question"],
    "target_answer": r["reference_answer"],
    "target_file_name": file_name(r),
    "target_page_no": r.get("source_pages"),
    "expected_source_ids": r["expected_source_ids"],
    "expected_behavior": r["expected_behavior"],
    "must_include": r["must_include"],
    "must_not_include": r["must_not_include"],
}

golden = [to_golden(r, CONTEXT_TYPE[r["case_id"]]) for r in picked]
assert len(golden) == 20 and len({g["case_id"] for g in golden}) == 20


# 제공 180문항 전체 → golden_set_provided.jsonl
# 근거 유형은 위 20문항만 직접 붙였으므로 나머지는 자동으로 추정한다.
pages = {}
for line in open("04_extracted/pages.jsonl", encoding="utf-8"):
    p = json.loads(line)
    pages[(p["source"].split("/")[-1], p["page"])] = p

def numbers(text):
    return {n.replace(",", "") for n in re.findall(r"\d[\d,.]*\d", text)}

def guess_context_type(r):
    # ponytail: 답의 숫자가 본문엔 없고 표/그림에만 있으면 table/image, 아니면 paragraph. 숫자 없는 표 문항은 paragraph로 잘못 붙음
    if r["case_id"] in CONTEXT_TYPE:
        return CONTEXT_TYPE[r["case_id"]]
    if not r["expected_source_ids"] and not r.get("source_pages"):
        return "none"
    ps = [pages.get((file_name(r), n)) for n in r.get("source_pages") or []]
    ps = [p for p in ps if p]
    missing = numbers(r["reference_answer"]) - numbers(" ".join(p["text"] for p in ps).replace(",", ""))
    if missing & numbers(" ".join(t for p in ps for t in p["tables"])):
        return "table"
    if missing & numbers(" ".join(f["markdown"] for p in ps for f in p["figures"])):
        return "image"
    return "paragraph"

provided = [to_golden(r, guess_context_type(r)) for r in rows.values()]
assert len(provided) == 180

for path, data in (("05_golden_set/golden_set.jsonl", golden), ("05_golden_set/golden_set_provided.jsonl", provided)):
    with open(path, "w", encoding="utf-8") as f:
        for g in data:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"{path}: {len(data)}문항 저장")
