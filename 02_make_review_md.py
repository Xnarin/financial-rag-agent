import html
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import pdfplumber

BASE = Path(__file__).parent
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
    "| 문서 | 쪽수 | 표 | 거의 빈 쪽 | 비교 문서 |",
    "|---|---|---|---|---|",
    *index_rows,
    "",
]
(EXTRACT_DIR / "index.md").write_text("\n".join(index), encoding="utf-8")
print(f"{len(docs)}개 문서 → {REVIEW_DIR}")

# 브라우저로 바로 보는 뷰어 (왼쪽 문서 목록, 오른쪽 원문 캡처 | 추출 텍스트)
viewer_data = {
    source: [
        {"page": p["page"], "img": f"review/images/{Path(source).stem}/p{p['page']:03d}.png", "content": p["content"], "n_tables": len(p["tables"])}
        for p in pages
    ]
    for source, pages in docs.items()
}
VIEWER_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>추출 결과 비교</title>
<style>
body { margin: 0; display: flex; font-family: sans-serif; background: #fff; color: #111; }
nav { width: 260px; height: 100vh; overflow: auto; border-right: 1px solid #ccc; flex-shrink: 0; }
nav div { padding: 8px 12px; cursor: pointer; font-size: 13px; border-bottom: 1px solid #eee; word-break: break-all; }
nav div.on { background: #dbeafe; font-weight: bold; }
main { flex: 1; height: 100vh; overflow: auto; padding: 0 16px; }
.page { display: flex; gap: 12px; border-bottom: 2px solid #999; padding: 12px 0; }
.page img { width: 50%; border: 1px solid #ccc; align-self: flex-start; }
.page .out { width: 50%; font-size: 13px; background: #f7f7f7; padding: 8px; overflow-x: auto; }
.out pre { margin: 0; white-space: pre-wrap; font-family: inherit; }
.out table { border-collapse: collapse; margin: 8px 0; background: #fff; }
.out td, .out th { border: 1px solid #999; padding: 2px 6px; font-size: 12px; }
.out th { background: #e0e7ff; }
.empty { background: #fee2e2 !important; }
</style></head>
<body><nav id="nav"></nav><main id="main"></main>
<script>
const DATA = __DATA__;
const nav = document.getElementById("nav"), main = document.getElementById("main");
function el(tag, text) { const e = document.createElement(tag); e.textContent = text; return e; }
// 마크다운 표(| 로 시작하는 줄 묶음)는 HTML 표로, 나머지는 본문 그대로 보여준다.
function renderContent(out, content) {
  const lines = content.split("\n");
  let i = 0;
  while (i < lines.length) {
    if (lines[i].startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
      const table = document.createElement("table");
      rows.filter(r => !/^\|( --- \|)+$/.test(r)).forEach((r, ri) => {
        const tr = table.insertRow();
        r.slice(2, -2).split(" | ").forEach(c => tr.appendChild(el(ri ? "td" : "th", c.replaceAll("\\|", "|"))));
      });
      out.appendChild(table);
    } else {
      const text = [];
      while (i < lines.length && !lines[i].startsWith("|")) text.push(lines[i++]);
      if (text.join("").trim()) out.appendChild(el("pre", text.join("\n").trim()));
    }
  }
}
function show(source) {
  [...nav.children].forEach(d => d.classList.toggle("on", d.dataset.src === source));
  main.innerHTML = "<h2>" + source + "</h2>";
  for (const p of DATA[source]) {
    const div = document.createElement("div");
    div.className = "page";
    div.innerHTML = '<img loading="lazy" src="' + encodeURI(p.img) + '"><div class="out"></div>';
    const out = div.querySelector(".out");
    out.appendChild(el("b", p.page + "쪽" + (p.n_tables ? " · 표 " + p.n_tables + "개" : "")));
    renderContent(out, p.content || "(추출된 텍스트 없음)");
    if (p.content.trim().length < 30) out.classList.add("empty");
    main.appendChild(div);
  }
  main.scrollTop = 0;
}
for (const source in DATA) {
  const d = document.createElement("div");
  const nt = DATA[source].reduce((a, p) => a + p.n_tables, 0);
  d.textContent = source + " (" + DATA[source].length + "쪽, 표 " + nt + "개)";
  d.dataset.src = source;
  d.onclick = () => show(source);
  nav.appendChild(d);
}
show(Object.keys(DATA)[0]);
</script></body></html>
"""
viewer = VIEWER_HTML.replace("__DATA__", json.dumps(viewer_data, ensure_ascii=False).replace("</", "<\\/"))
(EXTRACT_DIR / "viewer.html").write_text(viewer, encoding="utf-8")
print(f"뷰어 → {EXTRACT_DIR / 'viewer.html'}")
