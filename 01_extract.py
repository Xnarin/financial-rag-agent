import json
import re
from collections import defaultdict
from pathlib import Path

import pdfplumber

BASE = Path(__file__).parent
DATA_DIR = BASE / "01_data"
OUT_PATH = BASE / "04_extracted" / "pages.jsonl"
OUT_PATH.parent.mkdir(exist_ok=True)
GEMINI_PATH = BASE / "04_extracted" / "gemini_tables.json"  # 03_gemini_tables.py 결과
IMAGE_PATH = BASE / "04_extracted" / "image_regions.json"  # 04_image_regions.py 결과
FIGURE_NAMES = {"chart": "그래프", "image": "그림", "table": "그림", "diagram": "그림", "formula": "수식"}
# 그 밖(OCR한 text 등)은 "이미지 속 글자"

# 폰트에 글자 정보가 없는 기호는 (cid:49049)처럼 나온다. 이 자료에서는 전부 화살표라서 →로 바꾼다.
CID = re.compile(r"\(cid:\d+\)")


def fix(text):
    # 깨진 기호 정리: (cid:…) → 화살표, Wingdings 글머리표(\uf09f 등) → •, Wingdings 공백(\uf020) → 공백
    text = CID.sub("→", text).replace("\uf020", " ")
    return re.sub(r"[\uf000-\uf0ff]", "•", text)


def clean_cell(cell):
    # 병합된 셀은 None으로 나온다. 셀 안 줄바꿈은 한 줄로 합친다.
    return " ".join(fix(cell or "").split()).replace("|", r"\|")


def table_to_markdown(rows):
    # 첫 행을 헤더로 쓰고, 행마다 열 개수를 헤더에 맞춘다.
    rows = [[clean_cell(c) for c in row] for row in rows]
    rows = [row for row in rows if any(row)]  # 완전히 빈 행은 버린다.
    n_cols = max((len(row) for row in rows), default=0)
    # 1행 또는 1열짜리는 표가 아니라 테두리 박스(목차·강조 문구)라서 본문으로 둔다.
    if len(rows) < 2 or n_cols < 2:
        return ""
    # 헤더 행·첫 열 말고는 다 빈 칸이면 표가 아니라 막대그래프 같은 그림이다.
    if not any(c for row in rows[1:] for c in row[1:]):
        return ""
    # 셀 하나가 1000자를 넘으면 표가 아니라 페이지 배경 상자다 (본문 전체가 한 칸에 들어감).
    if any(len(c) > 1000 for row in rows for c in row):
        return ""
    rows = [row + [""] * (n_cols - len(row)) for row in rows]
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    lines.insert(1, "| " + " | ".join(["---"] * n_cols) + " |")
    return "\n".join(lines)


def dotted_lines(page):
    # 점선은 길이 0짜리 점 수백 개로 그려져 있어서 pdfplumber가 선으로 안 본다.
    # 같은 줄에 6pt 이내 간격으로 10개 이상 늘어선 점들을 선 하나로 잇는다.
    # 가로 점선은 표의 행 구분선으로 쓴다.
    # 세로 점선은 강조 상자의 옆변이면 가짜 열이 생기므로 표 찾기에는 안 쓰고,
    # 첫 열의 들여쓰기 괄호(계층 표시)를 찾는 데만 쓴다.
    dots = [o for o in page.lines if o["x1"] - o["x0"] <= 2 and o["bottom"] - o["top"] <= 2]
    h_lines, v_lines = [], []
    for axis, pos, out in (("top", "x0", h_lines), ("x0", "top", v_lines)):
        groups = defaultdict(list)
        for d in dots:
            groups[round(d[axis])].append(d[pos])
        for key, values in groups.items():
            values.sort()
            run = [values[0]]
            for v in values[1:] + [None]:
                if v is not None and v - run[-1] <= 6:
                    run.append(v)
                    continue
                if len(run) >= 10:
                    if out is h_lines:
                        x0, x1, top, bottom = run[0], run[-1], key, key
                    else:
                        x0, x1, top, bottom = key, key, run[0], run[-1]
                    out.append({"x0": x0, "x1": x1, "top": top, "bottom": bottom,
                                "width": x1 - x0, "height": bottom - top, "object_type": "line"})
                run = [v]
    return h_lines, v_lines


def snap_dotted(page, dotted, x0, x1):
    # 들여쓰기된 항목 아래 점선은 칸 중간에서 시작해서 행 구분선 역할을 못 한다.
    # 점선 양 끝을 같은 높이를 지나는 가장 가까운 세로선(없으면 표 끝)까지 늘린다.
    v_edges = [e for e in page.edges if e["orientation"] == "v" and e["height"] > 5]
    snapped = []
    for line in dotted:
        cross = [e["x0"] for e in v_edges if e["top"] - 1 <= line["top"] <= e["bottom"] + 1] + [x0, x1]
        a = max([x for x in cross if x <= line["x0"] + 1], default=x0)
        b = min([x for x in cross if x >= line["x1"] - 1], default=x1)
        snapped.append(dict(line, x0=a, x1=b, width=b - a))
    return snapped


def refind_table(page, table, dotted):
    # 바깥 세로 테두리가 없는 표가 많아서, 표 영역 양 끝에 세로선을 직접 지정해 다시 찾는다.
    x0, top, x1, bottom = table.bbox
    px0, ptop, px1, pbottom = page.bbox
    if x1 <= px0 or x0 >= px1 or bottom <= ptop or top >= pbottom:
        return table  # 페이지 밖에 있는 표(인쇄 영역 밖)는 그대로 둔다.
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "explicit_horizontal_lines": snap_dotted(page, dotted, x0, x1),
        "explicit_vertical_lines": [x0, x1],
    }
    # 테두리가 표 영역 경계에 딱 걸치면 소수점 차이로 잘려 나가서 1pt 넓혀 자른다.
    found = page.crop((x0 - 1, top - 1, x1 + 1, bottom + 1), strict=False).find_tables(settings)
    return max(found, key=lambda t: len(t.cells)) if found else table


def merge_split_tables(page, tables, dotted):
    # 화살표 줄처럼 세로선이 끊긴 곳에서 표가 위아래로 쪼개진다.
    # 바로 아래(40pt 이내)에 붙어 있고 폭 안에 들어오는 표들을 헤더 표에 합치고,
    # 헤더 행의 열 경계를 세로선으로 지정해 다시 찾는다.
    tables = sorted(tables, key=lambda t: t.bbox[1])
    result = []  # (표, 합쳤는지)
    i = 0
    while i < len(tables):
        head = tables[i]
        x0, top, x1, bottom = head.bbox
        j = i + 1
        while j < len(tables):
            nx0, ntop, nx1, nbottom = tables[j].bbox
            if nx0 >= x0 - 2 and nx1 <= x1 + 2 and ntop - bottom < 40:
                bottom = nbottom
                j += 1
            else:
                break
        cols = sorted({round(x, 1) for c in head.rows[0].cells if c for x in (c[0], c[2])})
        if j > i + 1 and len(cols) >= 3:  # 헤더가 2열 이상일 때만
            settings = {
                "vertical_strategy": "explicit",
                "horizontal_strategy": "lines",
                "explicit_vertical_lines": cols,
                "explicit_horizontal_lines": dotted,
            }
            found = page.crop((x0, top, x1, bottom), strict=False).find_tables(settings)
            if found:
                result.append((max(found, key=lambda t: len(t.cells)), True))
                i = j
                continue
        result.append((head, False))
        i += 1
    return result


def row_centers(table):
    # 행마다 가장 낮은 셀의 가운데 y (병합 셀은 높으니까)
    centers = []
    for r in table.rows:
        short = min((b for b in r.cells if b), key=lambda b: b[3] - b[1])
        centers.append((short[1] + short[3]) / 2)
    return centers


def fill_merged_cells(table):
    # 병합된 칸은 None으로 나온다. 그 칸을 덮는 원래 셀의 값으로 채운다.
    rows = table.extract()
    cells = [(cell, text) for r, row in zip(table.rows, rows) for cell, text in zip(r.cells, row) if cell]
    n_cols = len(table.rows[0].cells)
    # 열은 가장 좁은 셀, 행은 가장 낮은 셀로 칸의 중심을 잡는다 (병합 셀은 넓거나 높으니까).
    col_x = []
    for c in range(n_cols):
        col_cells = [r.cells[c] for r in table.rows if r.cells[c]]
        narrow = min(col_cells, key=lambda b: b[2] - b[0]) if col_cells else None
        col_x.append((narrow[0] + narrow[2]) / 2 if narrow else None)
    for cy, row in zip(row_centers(table), rows):
        for c, value in enumerate(row):
            if value is None and col_x[c] is not None:
                cx = col_x[c]
                row[c] = next(
                    (text for (x0, top, x1, bottom), text in cells if x0 <= cx <= x1 and top <= cy <= bottom),
                    None,
                )
    return rows


def tree_guides(h_lines, v_lines):
    # 계층 괄호는 세로 점선에서 가로 점선이 ㄴ자로 뻗어 나간다.
    # 가로 점선이 시작되지 않는 세로 점선은 칸 구분선이라 계층으로 보지 않는다.
    return [v for v in v_lines
            if any(abs(h["x0"] - v["x0"]) <= 3 and v["top"] - 1 <= h["top"] <= v["bottom"] + 1 for h in h_lines)]


def add_hierarchy(page, table, rows, guides):
    # 첫 열의 세로 점선 괄호(들여쓰기 계층 표시)를 보고 항목에 상위 항목 경로를 붙인다.
    # 예: 가계신용(A+B) > 가계대출(A) > 주택담보대출
    # - 행의 깊이 = 그 행을 세로로 지나는, 첫 열 안쪽 점선 괄호의 개수
    # - 괄호 왼쪽에 글자가 있으면(예: "증감 내역") 그 글자가 상위 항목,
    #   비어 있으면 괄호 바로 위 행이 상위 항목이다.
    first_col = [r.cells[0] for r in table.rows if r.cells[0]]
    if not first_col:
        return rows
    left, right = table.bbox[0], min(b[2] for b in first_col)
    guides = [g for g in guides if left < g["x0"] < right]
    if not guides:
        return rows

    def text_in(x0, x1, top, bottom):
        # 글자 중심이 영역 안에 있는 글자만 모아 한 줄로 만든다.
        region = page.filter(lambda o: o["object_type"] != "char" or (
            x0 - 1 <= (o["x0"] + o["x1"]) / 2 <= x1 + 1 and top <= (o["top"] + o["bottom"]) / 2 <= bottom))
        return " ".join((region.extract_text() or "").split())

    side_label = {id(g): text_in(left, g["x0"], g["top"], g["bottom"]) for g in guides}

    path = []
    for r, row in zip(table.rows, rows):
        short = min((b for b in r.cells if b), key=lambda b: b[3] - b[1])
        cy = (short[1] + short[3]) / 2
        covering = sorted((g for g in guides if g["top"] - 1 <= cy <= g["bottom"] + 1), key=lambda g: g["x0"])
        if not covering:
            path = [row[0] or ""]
        else:
            label = text_in(covering[-1]["x0"], right, short[1], short[3]) or row[0] or ""
            parents = [side_label[id(g)] or (path[i] if i < len(path) else "") for i, g in enumerate(covering)]
            path = parents + [label]
        row[0] = " > ".join(p for p in path if p)
    return rows


def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0


def outside(bboxes):
    # 글자 중심이 표 영역 안에 있으면 본문에서 뺀다.
    def keep(obj):
        if obj["object_type"] != "char":
            return True
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)
    return keep


def extract_page(page, gemini=(), figures=()):
    # Gemini로 다시 뽑은 표(숫자 검증 통과)를 먼저 넣는다. 이 쪽(펼친 페이지면 이 반쪽)에 글자가 있는 것만.
    # 아래 중복 제거 때문에 같은 자리의 pdfplumber 표는 자동으로 빠진다.
    tables = [
        {"bbox": tuple(g["bbox"]), "markdown": g["markdown"], "from": "gemini"}
        for g in gemini
        if any(g["bbox"][0] <= (c["x0"] + c["x1"]) / 2 <= g["bbox"][2]
               and g["bbox"][1] <= (c["top"] + c["bottom"]) / 2 <= g["bbox"][3] for c in page.chars)
    ]
    dotted, v_dotted = dotted_lines(page)
    guides = tree_guides(dotted, v_dotted)
    found = page.find_tables({"explicit_horizontal_lines": dotted})
    candidates = merge_split_tables(page, found, dotted)
    # 페이지 배경 상자도 표 후보로 잡혀서, 다시 찾으면 안쪽 표가 한 번 더 나온다.
    # 작은 후보부터 받고, 이미 받은 표와 절반 넘게 겹치면 중복으로 보고 버린다.
    candidates.sort(key=lambda c: area(c[0].bbox))
    for t, merged in candidates:
        if not merged:
            t = refind_table(page, t, dotted)
        if any(overlap(t.bbox, kept["bbox"]) > 0.5 * min(area(t.bbox), area(kept["bbox"])) for kept in tables):
            continue
        markdown = table_to_markdown(add_hierarchy(page, t, fill_merged_cells(t), guides))
        if markdown:
            tables.append({"bbox": t.bbox, "markdown": markdown, "from": "pdfplumber"})
    tables.sort(key=lambda t: t["bbox"][1])

    # 본문: 표 영역과 그림 영역을 뺀 나머지 글자
    # (그림 영역 글자 = 벡터로 그린 그래프의 축 숫자 등이라 본문에 흩어지면 노이즈)
    body_page = page.filter(outside([t["bbox"] for t in tables] + [f["bbox"] for f in figures]))
    body_lines = body_page.extract_text_lines(return_chars=False)

    # 읽는 순서(위→아래)대로 본문 줄, 표, 그림 해석을 합친다.
    figure_name = [FIGURE_NAMES.get(f["label"], "이미지 속 글자") for f in figures]
    blocks = [(line["top"], fix(line["text"])) for line in body_lines]
    blocks += [(t["bbox"][1], "\n" + t["markdown"] + "\n") for t in tables]
    blocks += [(f["bbox"][1], f"\n[{name}]\n{f['markdown']}\n") for f, name in zip(figures, figure_name)]
    content = "\n".join(text for _, text in sorted(blocks, key=lambda b: b[0])).strip()

    return {
        "text": fix(body_page.extract_text() or ""),
        "tables": [t["markdown"] for t in tables],
        "table_sources": [t["from"] for t in tables],
        "figures": [{"type": name, "markdown": f["markdown"], "from": f["method"]}
                    for f, name in zip(figures, figure_name)],
        "content": content,
    }


def extract_spread(page, gemini=(), figures=()):
    # 두 쪽을 펼쳐서 한 장에 담은 가로로 긴 쪽은 왼쪽 반, 오른쪽 반을 따로 읽고 이어 붙인다.
    # 한 번에 읽으면 위→아래 순서로 정렬되면서 왼쪽 쪽과 오른쪽 쪽 문장이 섞인다.
    if page.width <= page.height:
        return extract_page(page, gemini, figures)
    # crop으로 자르면 그 안에서 표 영역을 다시 crop할 때 곡선 테두리가 사라져서,
    # 객체 가운데가 어느 쪽에 있는지로 나눈다.
    mid = (page.bbox[0] + page.bbox[2]) / 2
    halves = [
        extract_page(
            page.filter(lambda o: ((o["x0"] + o["x1"]) / 2 < mid) == is_left),
            gemini,
            [f for f in figures if ((f["bbox"][0] + f["bbox"][2]) / 2 < mid) == is_left],
        )
        for is_left in (True, False)
    ]
    return {
        "text": "\n".join(h["text"] for h in halves).strip(),
        "tables": [t for h in halves for t in h["tables"]],
        "table_sources": [s for h in halves for s in h["table_sources"]],
        "figures": [f for h in halves for f in h["figures"]],
        "content": "\n\n".join(h["content"] for h in halves).strip(),
    }


pdf_paths = sorted(DATA_DIR.rglob("*.pdf"))

gemini_tables = defaultdict(list)  # (문서, 쪽) → 숫자 검증을 통과한 Gemini 표
if GEMINI_PATH.exists():
    for g in json.loads(GEMINI_PATH.read_text(encoding="utf-8")):
        if g["status"] == "ok":
            gemini_tables[(g["source"], g["page"])].append(g)

def useful_figure(markdown):
    # 본문에 넣을 가치가 없는 그림 해석은 뺀다.
    # - "01", "22"처럼 장 번호 배지만 읽은 것 (글자 5개 미만)
    # - 값이 적혀 있지 않은 그래프의 제목만 남은 것 (예전 프롬프트 결과. 지금은 "설명:" 문장이 붙는다)
    if len(re.sub(r"\W", "", markdown)) < 5:
        return False
    return "설명:" in markdown or not ("값 표기 없음" in markdown and "|" not in markdown)


image_regions = defaultdict(list)  # (문서, 쪽) → 숫자 검증을 통과한 그림·그래프 해석
if IMAGE_PATH.exists():
    for r in json.loads(IMAGE_PATH.read_text(encoding="utf-8")):
        if r["status"] == "ok" and useful_figure(r["markdown"]):
            image_regions[(r["source"], r["page"])].append(r)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for pdf_path in pdf_paths:
        n_tables = n_figures = 0
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                source = str(pdf_path.relative_to(DATA_DIR)).replace("\\", "/")
                row = {"source": source, "page": i,
                       **extract_spread(page, gemini_tables[(source, i)], image_regions[(source, i)])}
                n_tables += len(row["tables"])
                n_figures += len(row["figures"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{pdf_path.name}: {len(pdf.pages)}쪽, 표 {n_tables}개, 그림 {n_figures}개")

print(f"저장: {OUT_PATH}")
