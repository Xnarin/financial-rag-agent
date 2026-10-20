"""그래프·그림·다이어그램·수식처럼 pdfplumber 글자만으로는 내용이 안 나오는 영역을 텍스트로 뽑는다.

1) 처리할 영역 찾기
   - rapid-layout으로 쪽마다 영역 분석 (결과는 04_extracted/layout.json에 저장해 재사용)
     · chart                  → 항상
     · image / table / 글자류 → 글자 정보(디지털 글자)가 없을 때만 (그림 파일)
   - 다이어그램: 작은 글상자 3개 이상이 가로로 나란히 이어진 줄 (벡터로 그린 절차도)
   - 깨진 수식: 수식 폰트 깨짐 문자, 분수 막대(―――), "="로 끝나는 줄
2) 영역 종류별 처리
   - chart / image / table / diagram → Gemini (설명 문장 + 표·흐름·목록, 장식이면 버림)
   - formula                         → Gemini (한 줄 수식 + 설명)
   - 글자류 (글자 정보 없음)          → RapidOCR
   Gemini가 쓴 숫자는 그 영역의 OCR 결과 + pdfplumber 글자와 대조해 검증한다.
   (이미지 속 table은 우리 문서에선 전부 앱·홈택스 화면 캡처라 rapid-table 대신 Gemini로 보낸다.)
3) 결과 → 04_extracted/image_regions.json (01_extract.py가 읽어서 본문에 넣는다)
"""
import base64
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pdfplumber
from rapid_layout import RapidLayout

BASE = Path(__file__).parent.parent  # 프로젝트 루트 (스크립트는 extraction/ 안에 있음)
DATA_DIR = BASE / "01_data"
LAYOUT_PATH = BASE / "04_extracted" / "layout.json"
GEMINI_TABLES_PATH = BASE / "04_extracted" / "gemini_tables.json"
OUT_PATH = BASE / "04_extracted" / "image_regions.json"

LAYOUT_DPI = 120
TEXT_LABELS = {"text", "content", "paragraph_title", "doc_title", "abstract"}
PAD = 10  # 레이아웃 상자가 문서에 따라 최대 ~10pt 어긋나서(insurance) 여유를 둔다
PROMPT_VERSION = 2  # 프롬프트를 바꾸면 올린다 → 예전 버전 Gemini 결과만 다시 받는다


def crop_offset(page):
    # to_image()는 실제 보이는 영역(cropbox)만 그린다. 이미지 (0, 0)이 pdfplumber 좌표로는 여기다.
    # cropbox·mediabox는 PDF 좌표(아래가 0)라 위쪽 여백은 mediabox 위끝 - cropbox 위끝.
    return page.cropbox[0] - page.mediabox[0], page.mediabox[3] - page.cropbox[3]


def run_layout():
    # 490쪽 레이아웃 분석은 십여 분 걸려서 한 번 돌린 결과를 저장해 두고 쓴다.
    if LAYOUT_PATH.exists():
        return json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    layout = RapidLayout(model_type="pp_doc_layoutv3")
    regions = []
    for pdf_path in sorted(DATA_DIR.rglob("*.pdf")):
        source = str(pdf_path.relative_to(DATA_DIR)).replace("\\", "/")
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                img = np.array(page.to_image(resolution=LAYOUT_DPI).original.convert("RGB"))
                out = layout(img)
                scale = 72 / LAYOUT_DPI  # 이미지 픽셀 → PDF 좌표
                dx, dy = crop_offset(page)
                for box, label, score in zip(out.boxes or [], out.class_names or [], out.scores or []):
                    regions.append({
                        "source": source, "page": i, "label": label, "score": round(float(score), 3),
                        "bbox": [round(v * scale + (dx if k % 2 == 0 else dy), 1) for k, v in enumerate(box)],
                    })
        print(f"레이아웃: {source}")
    LAYOUT_PATH.write_text(json.dumps(regions, ensure_ascii=False), encoding="utf-8")
    return regions


def char_count(page, bbox):
    # 영역(+여유) 안에 pdfplumber 글자(디지털 글자)가 몇 개 있는지
    x0, top, x1, bottom = bbox[0] - PAD, bbox[1] - PAD, bbox[2] + PAD, bbox[3] + PAD
    return sum(x0 <= (c["x0"] + c["x1"]) / 2 <= x1 and top <= (c["top"] + c["bottom"]) / 2 <= bottom
               for c in page.chars)


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def overlap_ratio(a, b):
    # a 넓이 중 b와 겹치는 비율
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return (w * h) / ((a[2] - a[0]) * (a[3] - a[1])) if w > 0 and h > 0 else 0


def find_diagrams(regions):
    # 벡터로 그린 절차도는 레이아웃 모델이 상자 하나하나를 작은 text로 본다.
    # 작은 글상자 3개 이상이 가로로 겹치지 않고 60pt 이내 간격으로 이어지면 다이어그램 한 줄로 본다.
    by_page = defaultdict(list)
    for r in regions:
        by_page[(r["source"], r["page"])].append(r)
    diagrams = []
    for (source, page_no), rs in by_page.items():
        small = [r["bbox"] for r in rs if r["label"] in TEXT_LABELS
                 and r["bbox"][2] - r["bbox"][0] < 130 and r["bbox"][3] - r["bbox"][1] < 70]
        rows, used = [], set()
        for a in small:
            if tuple(a) in used:
                continue
            cy = (a[1] + a[3]) / 2
            same_row = sorted((b for b in small if abs((b[1] + b[3]) / 2 - cy) < 20), key=lambda b: b[0])
            chain = [same_row[0]]
            for b in same_row[1:]:
                gap = b[0] - chain[-1][2]
                if -2 <= gap <= 60:
                    chain.append(b)
                elif len(chain) < 3:
                    chain = [b]
            if len(chain) >= 3:
                rows.append(union(chain))
                used |= {tuple(b) for b in chain}
        # 위아래로 30pt 이내에 붙은 줄은 한 다이어그램으로 합친다
        rows.sort(key=lambda b: b[1])
        merged = []
        for row in rows:
            if merged and row[1] - merged[-1][3] < 30 and row[0] < merged[-1][2] and row[2] > merged[-1][0]:
                merged[-1] = union([merged[-1], row])
            else:
                merged.append(row)
        # 바로 위(35pt 이내)의 제목(※ … 절차 등)도 포함한다
        for box in merged:
            titles = [r["bbox"] for r in rs if r["label"] in ("paragraph_title", "figure_title")
                      and box[1] - 35 <= r["bbox"][3] <= box[1] + 2 and r["bbox"][0] < box[2] and r["bbox"][2] > box[0]]
            diagrams.append({"source": source, "page": page_no, "label": "diagram", "score": 1.0,
                             "bbox": [round(v, 1) for v in union([box] + titles)]})
    return diagrams


def is_formula_line(text):
    # 수식 폰트 깨짐 문자(한 줄에 2개 이상, 1개는 화살표 아이콘), 분수 막대, "="로 끝나는 줄 (URL 제외)
    return "http" not in text and (len(re.findall(r"[\ue000-\uefff]", text)) >= 2
                                   or re.search(r"―{3,}", text) or re.search(r"[=＝]\s*$", text))


def find_formulas(fixed_boxes):
    # 깨진 수식 줄을 찾아 위아래 40pt 이내 줄끼리 묶는다. 분자·분모가 위아래에 있어서 위아래로 25pt 넓힌다.
    formulas = []
    for pdf_path in sorted(DATA_DIR.rglob("*.pdf")):
        source = str(pdf_path.relative_to(DATA_DIR)).replace("\\", "/")
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                lines = page.extract_text_lines()
                hits = [l for l in lines if is_formula_line(l["text"])]
                groups = []
                for l in sorted(hits, key=lambda l: l["top"]):
                    if groups and l["top"] - groups[-1][-1]["bottom"] < 40:
                        groups[-1].append(l)
                    else:
                        groups.append([l])
                for g in groups:
                    # 가로는 그 줄들이 속한 본문 열 전체 (그림 수식이 줄보다 넓은 경우가 있어서)
                    # 펼친 페이지는 그 줄이 있는 반쪽 안으로 자른다.
                    left, right = page.bbox[0], page.bbox[2]
                    if page.width > page.height:
                        mid = (left + right) / 2
                        left, right = (left, mid) if g[0]["x0"] < mid else (mid, right)
                    col = [l for l in lines if left <= l["x0"] < right]
                    box = [max(min(l["x0"] for l in col) - 5, left), min(l["top"] for l in g) - 25,
                           min(max(l["x1"] for l in col) + 5, right), max(l["bottom"] for l in g) + 25]
                    box = [round(v, 1) for v in box]
                    if any(overlap_ratio(f["bbox"], box) > 0.3 for f in fixed_boxes.get((source, i), [])):
                        continue  # 03_gemini_tables.py가 이미 고친 영역 (주택연금 105쪽 등)
                    formulas.append({"source": source, "page": i, "label": "formula", "score": 1.0, "bbox": box})
    return formulas


def pick_targets(regions):
    # 처리할 영역만 고른다: chart는 항상, 나머지 레이아웃 영역은 글자 정보가 없을 때만 + 다이어그램 + 수식.
    targets = []
    pages = {}
    for r in regions:
        if r["label"] not in {"chart", "image", "table"} | TEXT_LABELS:
            continue
        key = (r["source"], r["page"])
        if key not in pages:
            pages[key] = pdfplumber.open(DATA_DIR / r["source"]).pages[r["page"] - 1]
        n_chars = char_count(pages[key], r["bbox"])
        # 글자류 영역은 "출자전환"처럼 4글자 제목도 있어서, 글자 정보가 하나도 없을 때만 OCR한다.
        # 그림·표 영역은 캡션 글자가 몇 개 걸칠 수 있어서 5개 미만까지 허용한다.
        no_text = n_chars == 0 if r["label"] in TEXT_LABELS else n_chars < 5
        if r["label"] == "chart" or no_text:
            targets.append({**r, "n_chars": n_chars})

    fixed = defaultdict(list)
    if GEMINI_TABLES_PATH.exists():
        for g in json.loads(GEMINI_TABLES_PATH.read_text(encoding="utf-8")):
            fixed[(g["source"], g["page"])].append(g)
    extra = find_diagrams(regions) + find_formulas(fixed)
    # 다이어그램·수식 영역 안에 들어가는 레이아웃 영역은 빼고 큰 영역으로 한 번에 처리한다.
    targets = [t for t in targets
               if not any(e["source"] == t["source"] and e["page"] == t["page"]
                          and overlap_ratio(t["bbox"], e["bbox"]) > 0.5 for e in extra)]
    return targets + extra


FIGURE_PROMPT = """금융 문서에서 잘라낸 그림입니다. 아래 형식으로 옮겨 주세요.

설명: 그림이 무엇을 보여주는지 1~3문장으로 쓴다. 그래프면 제목·기간·단위와 주요 추세(증가·감소, 가장 높은·낮은 항목 등), 절차도면 전체 흐름을 문장으로 쓴다. 그림에 적힌 숫자만 쓰고, 계산하거나 어림하지 않는다.

(빈 줄 다음에 종류별 내용)
- 그래프: 막대·점 위에 적힌 값만 마크다운 표로 쓴다 (예: | 구분 | 2022 | 2023 |). 적힌 값이 없으면 표는 생략한다.
- 절차도·다이어그램: "1. A → 2. B"처럼 번호 목록으로, 상자 안 글자는 그대로 쓴다.
- 화면 캡처: 보이는 메뉴·항목 글자를 목록으로 쓴다.
- 글자가 들어 있는 그림(안내문 등): 보이는 글자를 줄 단위로 그대로 쓴다.
- 그림 제목, 주, 자료 출처가 함께 보이면 마지막에 그대로 쓴다.

사진·일러스트·아이콘, 장·쪽 번호 배지처럼 정보가 없는 장식이면 "장식" 한 단어만 쓴다.
이미지에 보이는 글자와 숫자만 쓰고 추측하지 않는다. 코드블록(```)으로 감싸지 않는다."""

FORMULA_PROMPT = """금융 문서에서 잘라낸 수식 영역입니다. 아래 형식으로 옮겨 주세요.

- 수식은 한 줄 텍스트로 쓴다: 분수는 (분자) / (분모), 거듭제곱은 ^, 곱셈은 ×, max·min은 max[a, min[b, c]] 형태.
- 수식 앞뒤의 정의 줄(예: *가용소득 = …)도 같은 방식으로 한 줄씩 쓴다.
- 그림으로 된 계산 예시는 "1,000만원 − (500만원 + 5만원) = 495만원"처럼 식으로 쓴다.
- 마지막 줄에 "설명: "으로 시작해서 무엇을 계산하는 식인지 한 문장으로 쓴다.

이미지에 보이는 글자와 숫자만 쓰고 추측하지 않는다. 코드블록(```)으로 감싸지 않는다."""

NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def numbers(text):
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def check_numbers(answer, reference):
    # Gemini가 쓴 숫자가 그림에 실제로 있는지 정답지(OCR 결과 + pdfplumber 글자)와 대조한다.
    # - OCR은 붙어 있는 막대 라벨을 한 덩어리로 읽어서(8,059 8,847 → 8,0598,847)
    #   정답지 숫자만 이어 붙인 문자열 안에 들어 있으면 통과로 본다.
    # - OCR이 끝자리를 놓치는 경우(63,587 → 63,58)도 있어서 3자리 이상 앞부분 일치도 인정한다.
    # - 1~2자리 숫자는 단계 번호 등이라 검증하지 않는다.
    # - 그림엔 '24.1처럼 적혀 있는데 설명 문장에 2024년으로 풀어 쓰는 경우가 많아서, 20xx는 뒤 두 자리만 확인한다.
    digits = " ".join(re.sub(r"\D", "", line) for line in reference.splitlines())
    ref = {n.replace(".", "") for n in numbers(reference)}
    unverified = []
    for n in numbers(answer):
        d = n.replace(".", "")
        if re.fullmatch(r"20\d\d", d):
            d = d[2:]  # 연도는 뒤 두 자리로 확인
        elif len(d) < 3:
            continue
        if d in digits or any(len(o) >= 3 and d.startswith(o) for o in ref):
            continue
        unverified.append(n)
    return sorted(unverified)


def process(targets):
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    load_dotenv(override=True)
    os.environ["LANGSMITH_TRACING"] = "false"  # 이미지 수백 장을 LangSmith에 올리지 않는다 (월 한도 초과로 오류만 남)
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=os.environ["GEMINI_API_KEY"], timeout=90, max_retries=2)
    korean = {"Rec.lang_type": LangRec.KOREAN, "Rec.model_type": ModelType.MOBILE,
              "Rec.ocr_version": OCRVersion.PPOCRV5, "Global.log_level": "error"}
    ocr = RapidOCR(params=korean)

    # 이미 처리한 영역은 다시 하지 않는다 (Gemini 결과가 매번 조금씩 달라서 고정).
    # 단, 예전 프롬프트로 받은 Gemini 결과는 다시 받는다 (장식 판정과 OCR 결과는 그대로 둔다).
    def keep(r):
        return r.get("version") == PROMPT_VERSION or r["status"] == "decoration" or r["method"] == "rapidocr"

    results = [r for r in json.loads(OUT_PATH.read_text(encoding="utf-8")) if keep(r)] if OUT_PATH.exists() else []
    done = {(r["source"], r["page"], tuple(r["bbox"])) for r in results}
    todo = [t for t in targets if (t["source"], t["page"], tuple(t["bbox"])) not in done]
    print(f"처리할 영역 {len(todo)}개 (이미 끝난 {len(results)}개는 건너뜀)")

    # PDF 렌더링(PDFium)과 OCR은 스레드 동시 사용이 안 돼서 한 번에 하나씩, Gemini 호출만 동시에 한다.
    lock = threading.Lock()

    def handle(t):
        with lock:
            with pdfplumber.open(DATA_DIR / t["source"]) as pdf:
                page = pdf.pages[t["page"] - 1]
                bbox = [max(t["bbox"][0] - PAD, page.bbox[0]), max(t["bbox"][1] - PAD, page.bbox[1]),
                        min(t["bbox"][2] + PAD, page.bbox[2]), min(t["bbox"][3] + PAD, page.bbox[3])]
                region = page.crop(bbox)
                pil = region.to_image(resolution=200).original.convert("RGB")
                pdf_text = region.extract_text() or ""
            ocr_out = ocr(np.array(pil))
        ocr_text = "\n".join(txt for txt, score in zip(ocr_out.txts or (), ocr_out.scores or ()) if score >= 0.8)

        if t["label"] in ("chart", "image", "table", "diagram", "formula"):
            buffer = BytesIO()
            pil.save(buffer, format="PNG")
            prompt = FORMULA_PROMPT if t["label"] == "formula" else FIGURE_PROMPT
            answer = llm.invoke([HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image", "base64": base64.b64encode(buffer.getvalue()).decode("ascii"), "mime_type": "image/png"},
            ])]).text.strip()
            if answer == "장식":
                result = {"method": "gemini", "status": "decoration", "markdown": ""}
            else:
                reference = ocr_text + "\n" + pdf_text  # 검증 정답지 (다시 검증할 때 OCR을 또 안 돌리도록 저장)
                unverified = check_numbers(answer, reference)
                result = {"method": "gemini", "status": "check" if unverified else "ok",
                          "markdown": answer, "unverified_numbers": unverified, "reference": reference}
        else:
            result = {"method": "rapidocr", "status": "ok" if len(ocr_text) >= 4 else "empty", "markdown": ocr_text}

        with lock:
            results.append({**t, "crop_bbox": bbox, "version": PROMPT_VERSION, **result})
            print(f"[{len(results)}/{len(targets)}] {result['status']:<10} {t['label']:<16} {t['source']} {t['page']}쪽")
            OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")  # 중간 저장

    # Gemini 응답을 기다리는 시간이 대부분이라 6개씩 동시에 보낸다.
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in [pool.submit(handle, t) for t in todo]:
            try:
                future.result()
            except Exception as e:  # 한 영역이 실패해도 나머지는 계속 (다시 돌리면 실패한 것만 재시도)
                print(f"실패: {e}")
    # 대상에서 빠진 예전 영역(다이어그램·수식 영역에 흡수된 것 등)은 결과에서도 지운다.
    wanted = {(t["source"], t["page"], tuple(t["bbox"])) for t in targets}
    results[:] = [r for r in results if (r["source"], r["page"], tuple(r["bbox"])) in wanted]
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    from collections import Counter

    regions = run_layout()
    targets = pick_targets(regions)
    print("처리 대상:", Counter(t["label"] for t in targets).most_common())
    if "--stats" in sys.argv:
        for t in targets:
            if t["label"] in ("diagram", "formula"):
                print(" ", t["label"], t["source"], t["page"], t["bbox"])
        sys.exit()
    results = process(targets)
    print("결과:", Counter((r["label"], r["status"]) for r in results).most_common())
