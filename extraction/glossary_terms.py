"""금융용어집 PDF → 용어 하나 = 한 줄 (04_extracted/glossary_terms.jsonl)

python extraction/glossary_terms.py

- 용어집은 "용어 한 줄(글자 크기 11) + 설명 몇 줄(글자 크기 10)"이 반복된다 → 글자 크기로 용어와 설명을 나눈다.
- 뒤쪽 "출처 목록" 쪽(용어 / 용어 설명 / URL?dicId=번호)은 정의가 없어서 청크로 쓰지 않고,
  용어별 번호(dicId)만 가져와 붙인다. 골든셋 정답 ID(fsc-glossary-1722)가 이 번호와 같다.
- chunk.py가 이 파일을 읽어서 용어집은 용어 하나 = 청크 하나로 만든다.
"""
import json
import re
from pathlib import Path

import pdfplumber

BASE = Path(__file__).parent.parent
PDF = BASE / "01_data" / "financial_glossary_readable.pdf"
PAGES_PATH = BASE / "04_extracted" / "pages.jsonl"
OUT_PATH = BASE / "04_extracted" / "glossary_terms.jsonl"
SOURCE = "financial_glossary_readable.pdf"


def norm(term):
    return re.sub(r"\s+", "", term)


def source_ids():
    # 출처 목록: "용어 / 용어 설명 / https://...dicId=1722..." → {용어: 1722}
    ids = {}
    for line in PAGES_PATH.open(encoding="utf-8"):
        p = json.loads(line)
        if p["source"] != SOURCE:
            continue
        for m in re.finditer(r"^(.+?) / 용어 설명 / \S*dicId=(\d+)", p["content"], re.M):
            ids[norm(m.group(1))] = m.group(2)
    return ids


def main():
    ids = source_ids()
    terms = []
    with pdfplumber.open(PDF) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            lines = {}  # 같은 높이의 글자 = 한 줄
            for ch in page.chars:
                lines.setdefault(round(ch["top"]), []).append(ch)
            for top in sorted(lines):
                chs = sorted(lines[top], key=lambda c: c["x0"])
                text = "".join(c["text"] for c in chs).strip()
                size = round(max(c["size"] for c in chs))
                if not text or "dicId=" in text or text in ("금융용어 모음", "출처 목록"):
                    continue  # 출처 목록·제목
                if size >= 11:  # 용어
                    terms.append({"term": text, "page": page_no, "lines": []})
                elif size == 10 and terms and terms[-1]["page"] == page_no:
                    terms[-1]["lines"].append(text)
                # 크기 8 이하 = 쪽 번호
    out = []
    for t in terms:
        if not t["lines"]:
            continue  # 설명 없는 줄 (출처 목록 쪽 머리 등)
        out.append({"term": t["term"], "definition": " ".join(t["lines"]), "page": t["page"],
                    "dic_id": ids.get(norm(t["term"]))})
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    missing = [t["term"] for t in out if not t["dic_id"]]
    print(f"용어 {len(out)}개 → {OUT_PATH.name} (출처 목록 번호 {len(ids)}개, 번호 못 찾은 용어 {len(missing)}개)")
    if missing:
        print("  번호 없음:", missing[:10])


if __name__ == "__main__":
    main()
