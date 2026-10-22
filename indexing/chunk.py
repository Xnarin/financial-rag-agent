"""04_extracted/pages.jsonl → 구조를 반영한 청크 (06_chunks/chunks_{크기}_{겹침}.jsonl)

비교 실험: 300/50 vs 500/60 (글자 수 기준). 두 설정 모두 같은 전처리를 쓰고 크기만 다르다.

- 머리글·꼬리말(쪽 번호, 반복 제목, 경고문 등)은 지운다.
- 표: 바로 위 제목·단위 줄(캡션 패턴 또는 문장이 아닌 짧은 줄, 최대 4줄) + 바로 아래 각주(주·자료·*·※)와 묶는다.
       청크 크기를 넘으면 행 단위로 나누고, 조각마다 제목과 헤더를 반복한다.
- 그림·그래프·수식 해석: 쪼개지 않고 통째로 한 청크 (설명 문장 포함).
- 본문: RecursiveCharacterTextSplitter로 쪽 단위로 나눈다 (출처 쪽 번호를 정확히 남기려고).
- 용어집: 용어 하나 = 청크 하나 ("[금융용어] 용어: 설명"), 뒤쪽 출처 목록 쪽은 뺀다.
  extraction/glossary_terms.py가 만든 04_extracted/glossary_terms.jsonl을 쓴다 (청크 크기 설정과 상관없음).
- 표준약관: 끝의 "출처 목록"(약관 이름 / 조항 제목 / URL) 줄은 뺀다. 청크 맨 앞에 "[약관 이름] 제○조(제목)" 머리말을 붙인다. 조항 중간에서 시작하는 청크도
  어느 약관 몇 조인지 알 수 있게 (쪽이 넘어가도 지금 조항을 기억). 메타데이터 article에도 저장.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = Path(__file__).parent.parent  # 프로젝트 루트 (스크립트는 indexing/ 안에 있음)
PAGES_PATH = BASE / "04_extracted" / "pages.jsonl"
GLOSSARY_PATH = BASE / "04_extracted" / "glossary_terms.jsonl"
GLOSSARY_SOURCE = "financial_glossary_readable.pdf"
OUT_DIR = BASE / "06_chunks"
CONFIGS = [(300, 50), (500, 60)]

FIGURE_TAGS = ("그래프", "그림", "수식", "이미지 속 글자")
NOTE = re.compile(r"^\s*(주\s*[:：0-9)]|자료\s*[:：]|<?\s*출처\s*[:：]|<\s*자료|\*|※|\(단위|<단위)")
ARTICLE = re.compile(r"^\s*(제\s*\d+\s*조(?:의\s*\d+)?\s*\([^)\n]{1,40}\))", re.M)  # 약관 조항 제목
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


def is_title(line):
    line = line.strip()
    # 앞 표의 각주(자료: …)는 제외
    return 0 < len(line) < 40 and not line.endswith(("다.", "다", "요.", "요", ".", "?", ":")) and not NOTE.search(line)


def attach_context(blocks):
    # 표·그림 바로 위의 제목·단위 줄과 바로 아래 각주 줄을 그 블록으로 옮긴다.
    out = []
    for i, (kind, text) in enumerate(blocks):
        if kind == "text":
            out.append([kind, text.splitlines(), [], []])
            continue
        before = out[-1][1] if out and out[-1][0] == "text" else []
        head = []
        # 제목 줄: 캡션 패턴이거나, 문장이 아닌 짧은 줄 ("현금흐름표(예시)", "20◯◯년 1월 1일 ~ 12월 31일", "유입 유출")
        while before and (not before[-1].strip() or (CAPTION.search(before[-1]) and len(before[-1]) < 80)
                          or is_title(before[-1])) and len(head) < 4:
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
    if page["source"].startswith("standard_terms/"):
        # 약관 끝의 "출처 목록"(약관 이름 / 조항 제목 / URL)은 내용 없이 조항 제목만 있어서 검색을 가로챈다 (E13)
        page["content"] = "\n".join(l for l in page["content"].splitlines()
                                    if l.strip() != "출처 목록" and "law.kofia" not in l and "Seq=" not in l)
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


def add_article_header(source, kind, text, extra, article):
    # 약관 청크 앞에 "[약관 이름] 제○조(제목)". 청크가 조항 제목으로 시작하면 약관 이름만 붙인다
    name = re.sub(r"^\d+_|\.pdf$", "", Path(source).name)
    heads = ARTICLE.findall(text)
    starts = bool(ARTICLE.match(text))
    current = heads[0] if starts else article.get(source, "")
    header = f"[{name}]" if starts or not current else f"[{name}] {current}"
    if heads:
        article[source] = heads[-1]
    return kind, f"{header}\n{text}", {**(extra or {}), "article": current, "header": header}


def main():
    pages = [json.loads(line) for line in PAGES_PATH.open(encoding="utf-8")]
    headers_footers = find_headers_footers(pages)
    glossary = defaultdict(list)  # 쪽 → [(용어, 설명, 번호)]
    for t in map(json.loads, GLOSSARY_PATH.open(encoding="utf-8")):
        glossary[t["page"]].append(t)
    OUT_DIR.mkdir(exist_ok=True)
    for size, overlap in CONFIGS:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        name = f"chunks_{size}_{overlap}"
        rows = []
        article = {}  # 약관 문서 → 지금 읽고 있는 조항 제목 (쪽이 넘어가도 이어짐)
        for page in pages:
            if page["source"] == GLOSSARY_SOURCE:
                # 용어 하나 = 청크 하나. 출처 목록 쪽(용어가 없는 쪽)은 청크를 안 만든다
                chunks = [("text", f"[금융용어] {t['term']}: {t['definition']}", t) for t in glossary[page["page"]]]
            else:
                chunks = [(kind, text, None) for kind, text in chunk_page(page, headers_footers[page["source"]], splitter, size)]
            if page["source"].startswith("standard_terms/"):
                chunks = [add_article_header(page["source"], kind, text, term, article) for kind, text, term in chunks]
            for n, (kind, text, term) in enumerate(chunks):
                rows.append({
                    "chunk_id": f"{size}_{overlap}:{page['source']}:p{page['page']}:{n}",
                    "source": page["source"],
                    "page": page["page"],
                    "doc_group": doc_group(page["source"]),
                    "block_type": kind,  # text / table / figure
                    "text": text,
                    **({"term": term["term"], "dic_id": term["dic_id"]} if term and "dic_id" in term else {}),
                    **({"article": term["article"]} if term and term.get("article") else {}),
                    **({"header": term["header"]} if term and term.get("header") else {}),  # BM25에서는 뺀다 (search.bm25_docs)
                })
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        lengths = sorted(len(r["text"]) for r in rows)
        kinds = Counter(r["block_type"] for r in rows)
        print(f"{name}: 청크 {len(rows)}개 {dict(kinds)} | 길이 중앙값 {lengths[len(lengths) // 2]}, 최대 {lengths[-1]}")


if __name__ == "__main__":
    main()
