"""pdfplumber로 제대로 안 뽑히는 표를 Gemini로 다시 뽑는다.

표 영역 캡처 → Gemini → 숫자 검증 → 04_extracted/gemini_tables.json
검증을 통과한 표는 01_extract.py가 읽어서 pdfplumber 표 대신 쓴다.
"""
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

BASE = Path(__file__).parent.parent  # 프로젝트 루트 (스크립트는 extraction/ 안에 있음)
DATA_DIR = BASE / "01_data"
OUT_PATH = BASE / "04_extracted" / "gemini_tables.json"

# 다시 뽑을 표: 문서, 쪽, 표 영역(x0, top, x1, bottom) — 좌표는 pdfplumber 기준
TARGETS = [
    {"source": "additional/insurance_terms_example_2020.pdf", "page": 5, "bbox": [140, 185, 515, 285]},
    {"source": "additional/housing_pension_2026.pdf", "page": 105, "bbox": [84, 220, 395, 284]},
    {"source": "additional/housing_pension_2026.pdf", "page": 27, "bbox": [86, 76, 397, 540]},
]

PROMPT = """이미지의 표를 마크다운 표로 옮겨 주세요.

규칙:
- 첫 행은 헤더, 둘째 행은 | --- | 구분선으로 쓴다.
- 병합된 셀은 그 셀이 덮는 모든 칸에 같은 값을 반복해서 쓴다.
- 셀 안의 줄바꿈은 공백 하나로 바꾼다.
- 수식은 한 줄 텍스트로 쓴다. 분수는 (분자/분모), 거듭제곱은 ^ (예: (1 + 주택가격 상승률/12)^t).
- 표 바깥 칸에 붙은 결과 표시(예: 재가입 불가)는 별도 열로 넣는다.
- 이미지에 보이는 글자와 숫자만 쓴다. 추측하거나 요약하거나 설명을 덧붙이지 않는다.
- 마크다운 표만 출력한다. 코드블록(```)으로 감싸지 않는다."""

NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def numbers(text):
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def check_numbers(markdown, region_text):
    # Gemini가 쓴 숫자가 원문 영역에 실제로 있는지 대조한다 (금융 문서라 지어낸 숫자는 치명적).
    found, source = numbers(markdown), numbers(region_text)
    return {"invented": sorted(found - source), "omitted": sorted(source - found)}


load_dotenv(override=True)  # 시스템 환경변수보다 .env 값을 우선
# GOOGLE_API_KEY가 있으면 그걸 먼저 쓰므로, Gemini 키를 직접 넘긴다.
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.environ["GEMINI_API_KEY"], timeout=90, max_retries=1)

# 이미 검증을 통과한 표는 다시 부르지 않는다 (호출마다 결과가 조금씩 달라서 한 번 통과한 걸 고정).
results = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else []
done = {(r["source"], r["page"], tuple(r["bbox"])) for r in results if r["status"] == "ok"}
results = [r for r in results if (r["source"], r["page"], tuple(r["bbox"])) in done]

for target in TARGETS:
    key = (target["source"], target["page"], tuple(target["bbox"]))
    if key in done:
        print(f"건너뜀(이미 통과): {target['source']} {target['page']}쪽")
        continue

    with pdfplumber.open(DATA_DIR / target["source"]) as pdf:
        region = pdf.pages[target["page"] - 1].crop(target["bbox"])
        region_text = region.extract_text() or ""
        buffer = BytesIO()
        region.to_image(resolution=200).original.save(buffer, format="PNG")

    message = HumanMessage(content=[
        {"type": "text", "text": PROMPT},
        {"type": "image", "base64": base64.b64encode(buffer.getvalue()).decode("ascii"), "mime_type": "image/png"},
    ])
    markdown = llm.invoke([message]).text.strip()

    check = check_numbers(markdown, region_text)
    status = "rejected" if check["invented"] else "ok"
    results.append({**target, "markdown": markdown, "status": status, **check})

    print(f"[{status}] {target['source']} {target['page']}쪽")
    if check["invented"]:
        print(f"  원문에 없는 숫자: {check['invented']}")
    if check["omitted"]:
        print(f"  표에 빠진 숫자(확인 필요): {check['omitted']}")

OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"저장: {OUT_PATH}")
