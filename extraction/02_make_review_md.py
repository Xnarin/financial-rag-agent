import html
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import pdfplumber

BASE = Path(__file__).parent.parent  # 프로젝트 루트 (스크립트는 extraction/ 안에 있음)
DATA_DIR = BASE / "01_data"
EXTRACT_DIR = BASE / "04_extracted"
REVIEW_DIR = EXTRACT_DIR / "review"
IMAGE_DIR = REVIEW_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 문서별로 페이지 모으기
docs = defaultdict(list)
for line in open(EXTRACT_DIR / "pages.jsonl", encoding="utf-8"):
    row = json.loads(line)
    docs[row["source"]].append(row)

index_rows = []
for source, pages in docs.items():
    stem = Path(source).stem
    (IMAGE_DIR / stem).mkdir(exist_ok=True)
    empty_pages = [p["page"] for p in pages if len(p["content"].strip()) < 30]
    n_tables = sum(len(p["tables"]) for p in pages)

    lines = [
        f"# {source}",
        "",
        f"- 쪽수: {len(pages)}",
        f"- 거의 빈 쪽: {empty_pages or '없음'}",
        f"- 표: {n_tables}개",
        "",
    ]
    with pdfplumber.open(DATA_DIR / source) as pdf:
        for p in pages:
            # 페이지 캡처 (review/images/<문서>/p001.png)
            img_name = f"p{p['page']:03d}.png"
            pdf.pages[p["page"] - 1].to_image(resolution=80).save(IMAGE_DIR / stem / img_name)
            img_src = quote(f"images/{stem}/{img_name}")

            # 왼쪽: 원문 캡처 / 오른쪽: 추출 결과 (본문 + 마크다운 표)
            text = html.escape(p["content"]) or "(추출된 텍스트 없음)"
            lines += [
                f"## {p['page']}쪽",
                "",
                "<table><tr>",
                f'<td width="50%" valign="top"><img src="{img_src}" width="100%"></td>',
                f'<td width="50%" valign="top"><pre style="white-space: pre-wrap">{text}</pre></td>',
                "</tr></table>",
                "",
            ]
    (REVIEW_DIR / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    index_rows.append(f"| {source} | {len(pages)} | {n_tables} | {len(empty_pages)} | [보기](review/{quote(stem)}.md) |")
    print(f"{stem}: {len(pages)}쪽")

index = [
    "# 추출 결과 검토",
    "",
    "> 그림 해석 중 숫자 검증을 못 통과해 본문에서 뺀 항목은 [image_check.md](image_check.md)에서 확인해요.",
    "",
    "| 문서 | 쪽수 | 표 | 거의 빈 쪽 | 비교 문서 |",
    "|---|---|---|---|---|",
    *index_rows,
    "",
]
(EXTRACT_DIR / "index.md").write_text("\n".join(index), encoding="utf-8")
print(f"{len(docs)}개 문서 → {REVIEW_DIR}")

# 그림 해석 중 숫자 검증을 못 통과한 것(check) 검토 목록
# 맞으면 image_regions.json에서 그 항목의 status를 "ok"로 바꾸고 01_extract.py를 다시 돌린다.
IMAGE_REGIONS = EXTRACT_DIR / "image_regions.json"
if IMAGE_REGIONS.exists():
    checks = [r for r in json.loads(IMAGE_REGIONS.read_text(encoding="utf-8")) if r["status"] == "check"]
    (IMAGE_DIR / "checks").mkdir(exist_ok=True)
    lines = [
        "# 그림 해석 검토 목록",
        "",
        "Gemini가 쓴 숫자 중 OCR로 그림에서 찾지 못한 것이 있어 **본문에 넣지 않은** 항목이에요.",
        "원본과 비교해서 맞으면 `image_regions.json`에서 그 항목의 `status`를 `\"ok\"`로 바꾸고 `01_extract.py`를 다시 돌려요.",
        "",
    ]
    for n, r in enumerate(checks, start=1):
        img_name = f"check_{n:02d}.png"
        with pdfplumber.open(DATA_DIR / r["source"]) as pdf:
            pdf.pages[r["page"] - 1].crop(r["crop_bbox"]).to_image(resolution=110).save(IMAGE_DIR / "checks" / img_name)
        lines += [
            f"## {n}. {r['source']} {r['page']}쪽 ({r['label']})",
            "",
            f"- 그림에서 찾지 못한 숫자: `{', '.join(r['unverified_numbers'])}`",
            f"- bbox: `{r['bbox']}`",
            "",
            "<table><tr>",
            f'<td width="50%" valign="top"><img src="review/images/checks/{img_name}" width="100%"></td>',
            f'<td width="50%" valign="top"><pre style="white-space: pre-wrap">{html.escape(r["markdown"])}</pre></td>',
            "</tr></table>",
            "",
        ]
    (EXTRACT_DIR / "image_check.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"검토 목록: {len(checks)}개 → {EXTRACT_DIR / 'image_check.md'}")
