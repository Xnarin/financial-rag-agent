"""04_extracted/pages.jsonl → 구조를 반영한 청크 (06_chunks/chunks_{크기}_{겹침}.jsonl)

비교 실험: 300/50 vs 500/60 (글자 수 기준). 두 설정 모두 같은 전처리를 쓰고 크기만 다르다.

- 머리글·꼬리말(쪽 번호, 반복 제목, 경고문 등)은 지운다.
- 표: 바로 위 제목·단위 줄 + 바로 아래 각주(주·자료·*·※)와 묶는다.
       청크 크기를 넘으면 행 단위로 나누고, 조각마다 제목과 헤더를 반복한다.
- 그림·그래프·수식 해석: 쪼개지 않고 통째로 한 청크 (설명 문장 포함).
- 본문: RecursiveCharacterTextSplitter로 쪽 단위로 나눈다 (출처 쪽 번호를 정확히 남기려고).
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = Path(__file__).parent.parent  # 프로젝트 루트 (스크립트는 indexing/ 안에 있음)
PAGES_PATH = BASE / "04_extracted" / "pages.jsonl"
OUT_DIR = BASE / "06_chunks"
CONFIGS = [(300, 50), (500, 60)]

FIGURE_TAGS = ("그래프", "그림", "수식", "이미지 속 글자")
NOTE = re.compile(r"^\s*(주\s*[:：0-9)]|자료\s*[:：]|<?\s*출처\s*[:：]|<\s*자료|\*|※|\(단위|<단위)")
CAPTION = re.compile(r"^\s*(\[|<|〔)?\s*(표|그림|그래프)\s*[\dⅠ-Ⅹ]|\(단위|<단위|단위\s*[:：]")


def doc_group(source):
    # 과제의 질문 세트별 검색 범위를 나누려고 붙이는 그룹 (기본 negative 세트는 용어집+약관만 검색)
    if source.startswith("standard_terms/"):
        return "standard_terms"
    if source.startswith("additional/"):
        return "additional"
    return "glossary" if "glossary" in source else "life_guide"


def find_headers_footers(pages):
    # 같은 문서에서 쪽 맨 위·아래 2줄에 숫자만 다르고 반복되는 줄 = 머리글·꼬리말
    counts = defaultdict(Counter)
    for p in pages:
        lines = [l.strip() for l in p["content"].splitlines() if l.strip() and not l.startswith("|")]
        for l in lines[:2] + lines[-2:]:
            counts[p["source"]][re.sub(r"\d+", "#", l)] += 1
    n_pages = Counter(p["source"] for p in pages)
    return {source: {k for k, v in c.items() if v >= max(3, n_pages[source] * 0.15)}
            for source, c in counts.items()}


def remove_headers_footers(lines, patterns):
    text_idx = [i for i, l in enumerate(lines) if l.strip() and not l.startswith("|")]
    edge = set(text_idx[:2] + text_idx[-2:])
    return [l for i, l in enumerate(lines) if not (i in edge and re.sub(r"\d+", "#", l.strip()) in patterns)]


def split_blocks(page):
    # content를 본문 / 표 / 그림 블록으로 나눈다. 표·그림 문자열은 tables·figures에 정확히 있으니 위치로 자른다.
    content = page["content"]
    spans = []
    for t in page["tables"]:
        i = content.find(t)
        spans.append((i, i + len(t), "table", t))
    for f in page["figures"]:
        block = f"[{f['type']}]\n{f['markdown']}"
        i = content.find(block)
        spans.append((i, i + len(block), "figure", block))
    blocks, pos = [], 0
    for start, end, kind, text in sorted(s for s in spans if s[0] >= 0):
        if start < pos:
            continue
        blocks.append(("text", content[pos:start]))
        blocks.append((kind, text))
        pos = end
    blocks.append(("text", content[pos:]))
    return blocks


def attach_context(blocks):
    # 표·그림 바로 위의 제목·단위 줄과 바로 아래 각주 줄을 그 블록으로 옮긴다.
    out = []
    for i, (kind, text) in enumerate(blocks):
        if kind == "text":
            out.append([kind, text.splitlines(), [], []])
            continue
        before = out[-1][1] if out and out[-1][0] == "text" else []
        head = []
        while before and (not before[-1].strip() or (CAPTION.search(before[-1]) and len(before[-1]) < 80)) and len(head) < 3:
            line = before.pop()
            if line.strip():
                head.insert(0, line.strip())
        # 앞 본문이 80자 미만이면 "신탁방식과 저당권 방식의 차이점(비교표)" 같은 제목뿐이라 통째로 붙인다
        if before and len("".join(before).strip()) < 80:
            head = [l.strip() for l in before if l.strip()] + head
            before.clear()
        out.append([kind, text.splitlines(), head, []])
    # 각주는 다음 본문 블록 앞부분에서 가져온다
    for i, block in enumerate(out):
        if block[0] == "text" or i + 1 >= len(out) or out[i + 1][0] != "text":
            continue
        after = out[i + 1][1]
        while after and (not after[0].strip() or NOTE.search(after[0])):
            line = after.pop(0)
            if line.strip():
                block[3].append(line.strip())
    return out


def table_chunks(lines, head, notes, size):
    # 표가 크면 행 단위로 나누고 조각마다 제목·헤더를 반복한다.
    header, rows = lines[:2], lines[2:]
    prefix = "\n".join(head + header)
    suffix = "\n".join(notes)
    whole = "\n".join(head + lines + notes)
    if len(whole) <= size or not rows:
        return [whole]
    chunks, current = [], []
    for row in rows:
        if current and len(prefix) + len("\n".join(current + [row])) + 1 > size:
            chunks.append(current)
            current = []
        current.append(row)
    chunks.append(current)
    pieces = [prefix + "\n" + "\n".join(c) for c in chunks]
    if suffix:
        pieces[-1] += "\n" + suffix
    return pieces


def chunk_page(page, patterns, splitter, size):
    # 머리글·꼬리말을 먼저 지운다 (쪽 번호가 표 제목에 딸려 들어가지 않게)
    page = {**page, "content": "\n".join(remove_headers_footers(page["content"].splitlines(), patterns))}
    chunks = []
    for kind, lines, head, notes in attach_context(split_blocks(page)):
        if kind == "text":
            text = "\n".join(lines).strip()
            chunks += [("text", c) for c in splitter.split_text(text)] if text else []
        elif kind == "table":
            chunks += [("table", c) for c in table_chunks(lines, head, notes, size)]
        else:
            chunks.append(("figure", "\n".join(head + lines + notes)))
    return chunks


def main():
    pages = [json.loads(line) for line in PAGES_PATH.open(encoding="utf-8")]
    headers_footers = find_headers_footers(pages)
    OUT_DIR.mkdir(exist_ok=True)
    for size, overlap in CONFIGS:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        name = f"chunks_{size}_{overlap}"
        rows = []
        for page in pages:
            for n, (kind, text) in enumerate(chunk_page(page, headers_footers[page["source"]], splitter, size)):
                rows.append({
                    "chunk_id": f"{size}_{overlap}:{page['source']}:p{page['page']}:{n}",
                    "source": page["source"],
                    "page": page["page"],
                    "doc_group": doc_group(page["source"]),
                    "block_type": kind,  # text / table / figure
                    "text": text,
                })
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        lengths = sorted(len(r["text"]) for r in rows)
        kinds = Counter(r["block_type"] for r in rows)
        print(f"{name}: 청크 {len(rows)}개 {dict(kinds)} | 길이 중앙값 {lengths[len(lengths) // 2]}, 최대 {lengths[-1]}")


if __name__ == "__main__":
    main()
