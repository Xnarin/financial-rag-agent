"""검색 도구: 검색어 → 상위 k개 청크 + 청크 ID·출처(파일명·쪽).

python agent/search.py "주택연금은 무상 지원금인가요?"              # 확인용
python agent/search.py "임의매매 금지" --groups=glossary,standard_terms --k=3 --neighbors=0

확정한 설정 (README "검색 평가"·"답변 평가" 참고)
- 300/50 청크, Hybrid = 벡터(Chroma) + BM25(Kiwi 토큰화) 5:5, 각 후보 20개 → 상위 k=5
- 이웃 청크 ±1: 검색된 청크마다 같은 문서의 앞뒤 1개를 같이 돌려준다. 범위 끝이 표 조각이면 그 표의 나머지 조각까지
- groups: doc_group(glossary / standard_terms / life_guide / additional)으로 검색 범위 제한.
  과제의 기본 positive·negative 세트는 용어집 + 표준약관만 검색하는 조건 (BASIC_GROUPS)
"""
import json
import sys
from pathlib import Path

from kiwipiepy import Kiwi
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "indexing"))
from common import chroma_dir, get_embeddings  # noqa: E402

CHUNK_DIR = BASE / "06_chunks"
CONFIG = "chunks_300_50"
K, FETCH_K, WEIGHTS, NEIGHBORS = 5, 20, [0.5, 0.5], 1
BASIC_GROUPS = ["glossary", "standard_terms"]

kiwi = Kiwi()


def kiwi_tokenize(text):
    # BM25 토큰화 (강의 스크립트와 같음): 명사·동사·수식언·접사 + 외국어·숫자
    return [t.form.lower() for t in kiwi.tokenize(text)
            if t.tag.startswith(("N", "V", "M", "X")) or t.tag in {"SL", "SN"}]


def load_chunks(name=CONFIG):
    return [Document(page_content=c["text"], metadata=c, id=c["chunk_id"])
            for c in map(json.loads, (CHUNK_DIR / f"{name}.jsonl").open(encoding="utf-8"))]


def label(doc):
    return f"{Path(doc.metadata['source']).name} p{doc.metadata['page']}"


def to_context(docs):
    # 답변 LLM에 넘길 근거 글: 청크마다 [파일명 p쪽] 머리표
    return "\n\n".join(f"[{label(d)}]\n{d.page_content}" for d in docs)


def same_table(a, b):
    # 쪼개진 표 조각은 첫 줄(제목 또는 헤더)이 같다
    return (a.metadata["block_type"] == b.metadata["block_type"] == "table" and a.metadata["source"] == b.metadata["source"]
            and a.page_content.split("\n")[0] == b.page_content.split("\n")[0])


class SearchTool:
    def __init__(self, embeddings=None, config=CONFIG):
        self.embeddings = embeddings or get_embeddings()
        self.config = config
        self.vectorstore = Chroma(collection_name=config, embedding_function=self.embeddings,
                                  persist_directory=str(chroma_dir()))
        self.chunks = load_chunks(config)  # 파일 순서 = 문서·쪽·청크 순서 (이웃 찾기에 씀)
        self.index = {c.id: i for i, c in enumerate(self.chunks)}
        self.hybrids = {}  # 검색 범위(groups)마다 한 번만 만든다 (BM25 색인 만들기가 1~2초 걸림)

    def hybrid(self, groups=None):
        key = tuple(groups) if groups else None
        if key not in self.hybrids:
            chunks = [c for c in self.chunks if not groups or c.metadata["doc_group"] in groups]
            search_kwargs = {"k": FETCH_K} | ({"filter": {"doc_group": {"$in": list(groups)}}} if groups else {})
            bm25 = BM25Retriever.from_documents(chunks, preprocess_func=kiwi_tokenize, k=FETCH_K)
            self.hybrids[key] = EnsembleRetriever(
                retrievers=[self.vectorstore.as_retriever(search_kwargs=search_kwargs), bm25],
                weights=WEIGHTS, id_key="chunk_id")
        return self.hybrids[key]

    def with_neighbors(self, docs, n):
        # 검색된 청크마다 같은 문서의 앞뒤 n개를 문서 순서대로 붙인다 (중복 제외, 검색 순서 유지)
        # 범위 끝이 표 조각이면 그 표의 나머지 조각까지 넓힌다 (표 숫자가 조각마다 흩어져 있어서)
        out, seen = [], set()
        for d in docs:
            i = self.index[d.metadata["chunk_id"]]
            lo, hi = max(0, i - n), min(len(self.chunks) - 1, i + n)
            while lo > 0 and same_table(self.chunks[lo - 1], self.chunks[lo]):
                lo -= 1
            while hi < len(self.chunks) - 1 and same_table(self.chunks[hi], self.chunks[hi + 1]):
                hi += 1
            for j in range(lo, hi + 1):
                c = self.chunks[j]
                if c.metadata["source"] == d.metadata["source"] and c.id not in seen:
                    seen.add(c.id)
                    out.append(c)
        return out

    def search(self, query, k=K, neighbors=NEIGHBORS, groups=None):
        """상위 k개 + 이웃 청크를 돌려준다. metadata["rank"]: 검색 순위(1~k), 이웃으로 붙은 청크는 0."""
        top = self.hybrid(groups).invoke(query)[:k]
        ranks = {d.metadata["chunk_id"]: r for r, d in enumerate(top, 1)}
        docs = self.with_neighbors(top, neighbors) if neighbors else top
        # 원본 청크(self.chunks)를 건드리지 않게 복사해서 순위를 붙인다
        return [Document(page_content=d.page_content, id=d.metadata["chunk_id"],
                         metadata={k_: v for k_, v in d.metadata.items() if k_ != "text"}
                         | {"rank": ranks.get(d.metadata["chunk_id"], 0)})
                for d in docs]


def as_dicts(docs):
    # 도구 결과를 JSON으로 저장하거나 보여 줄 때: 청크 ID·출처·순위·본문
    return [{"chunk_id": d.metadata["chunk_id"], "source": Path(d.metadata["source"]).name,
             "page": d.metadata["page"], "rank": d.metadata["rank"], "text": d.page_content} for d in docs]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    tool = SearchTool()
    results = tool.search(" ".join(args) or "주택연금은 무상 지원금인가요?", k=int(opts.get("k", K)),
                          neighbors=int(opts.get("neighbors", NEIGHBORS)),
                          groups=opts["groups"].split(",") if "groups" in opts else None)
    for r in as_dicts(results):
        print(f"{'이웃' if r['rank'] == 0 else str(r['rank']) + '위':>4} | {r['source']} p{r['page']} | {r['chunk_id']}")
        print("       " + r["text"][:120].replace("\n", " / "))
