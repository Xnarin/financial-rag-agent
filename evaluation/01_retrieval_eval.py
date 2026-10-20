"""골든셋으로 검색 성능을 잰다.

python evaluation/01_retrieval_eval.py                       # 05_golden_set/*.jsonl 전부, 파일마다 따로
python evaluation/01_retrieval_eval.py 05_golden_set/xxx.jsonl  # 지정한 골든셋만

청킹 설정(300/50, 500/60) × 검색 방식(Similarity, MMR, BM25, Hybrid)을 난이도별·근거 유형별로 비교한다.
검색 설정은 강의 평가 스크립트(10-rag-evalution-script)와 같다: k=5, fetch_k=20, MMR λ=0.5, Hybrid 0.5:0.5, BM25는 Kiwi 토큰화.

정답(관련 청크) 기준
- target_file_name + target_page_no가 있으면: 그 파일의 그 쪽들 (쪽 단위)
- 약관 kofia-terms-{?}-{버전}-{조}: 버전 → 약관 파일(첫 쪽 "버전: 1785"), 조 → 그 조항이 시작하는 쪽
- 용어집 fsc-glossary-…: 쪽 정보가 없어서 파일 단위
- 정답 문서가 없는 문항(근거 없는 거절 문항 등)은 검색 평가에서 뺀다

지표 (질문마다 계산 후 평균)
- hit@k: 상위 k개 안에 관련 청크가 하나라도 있으면 1
- precision@5: 상위 5개 중 관련 청크 비율
- recall@k: 정답 쪽들 중 상위 k개가 덮은 비율 (파일 단위면 hit와 같음)
- mrr: 첫 관련 청크 순위의 역수
- latency_ms: 검색 한 번 걸린 시간. 질문 임베딩은 미리 한 번만 만들어 두므로 **순수 검색 시간**이다
  (질문 임베딩을 방식마다 다시 부르면 API 호출이 3배가 되고, 처음 부른 방식만 느려 보여서 불공정하다)

결과 → 07_eval/{골든셋 이름}/retrieval_{설정}.txt (난이도별), retrieval_{설정}_by_type.txt (근거 유형별),
       retrieval_rows.jsonl (질문별 상세)
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

from kiwipiepy import Kiwi
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "indexing"))
from common import COLLECTIONS, chroma_dir, get_embeddings  # noqa: E402

GOLDEN_DIR = BASE / "05_golden_set"
PAGES_PATH = BASE / "04_extracted" / "pages.jsonl"
CHUNK_DIR = BASE / "06_chunks"
OUT_DIR = BASE / "07_eval"

K, FETCH_K, LAMBDA_MULT, WEIGHTS = 5, 20, 0.5, [0.5, 0.5]
STRATEGIES = ["Similarity", "MMR", "BM25", "Hybrid"]
DIFFICULTY = {"basic": "easy", "intermediate": "medium", "advanced": "hard", "adversarial": "hard"}
GROUPS = {
    "level": ("easy", "medium", "hard"),                                       # 난이도별
    "context_type": ("paragraph", "table", "image", "formula", "none"),        # 근거 유형별 (있는 것만 표시)
}

kiwi = Kiwi()


def kiwi_tokenize(text):
    # 강의 스크립트와 같은 토큰화: 명사·동사·수식언·접사 + 외국어·숫자
    return [t.form.lower() for t in kiwi.tokenize(text)
            if t.tag.startswith(("N", "V", "M", "X")) or t.tag in {"SL", "SN"}]


class QueryCache(Embeddings):
    # 같은 질문의 임베딩을 한 번만 만든다 (Similarity·MMR·Hybrid가 같은 질문 벡터를 같이 씀)
    def __init__(self, base):
        self.base, self.cache = base, {}

    def embed_documents(self, texts):
        return self.base.embed_documents(texts)

    def embed_query(self, text):
        if text not in self.cache:
            for attempt in range(5):  # 속도 제한(429)이면 기다렸다 다시
                try:
                    self.cache[text] = self.base.embed_query(text)
                    break
                except Exception as e:
                    print(f"  질문 임베딩 재시도 {attempt + 1}: {str(e)[:80]}")
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError("질문 임베딩 실패")
        return self.cache[text]


def build_targets(golden):
    # 문항마다 정답을 (파일 이름, 쪽 집합 또는 None=파일 단위)으로 만든다.
    pages = [json.loads(line) for line in PAGES_PATH.open(encoding="utf-8")]
    version_file = {}
    for p in pages:
        m = re.search(r"버전\s*:\s*(\d+)", p["content"])
        if m and p["source"].startswith("standard_terms/") and p["page"] == 1:
            version_file[m.group(1)] = Path(p["source"]).name

    def article_page(file_name, article_no):
        # "제6조(" 로 시작하는 줄이 처음 나오는 쪽
        pattern = re.compile(rf"^\s*제\s*{article_no}\s*조\s*\(", re.M)
        for p in pages:
            if Path(p["source"]).name == file_name and pattern.search(p["content"]):
                return p["page"]
        return None

    targets = {}
    for g in golden:
        if g.get("target_file_name") and g.get("target_page_no"):
            targets[g["case_id"]] = {(g["target_file_name"], page) for page in g["target_page_no"]}
        elif g.get("target_file_name"):
            targets[g["case_id"]] = {(g["target_file_name"], None)}  # 파일 단위
        else:
            found = set()
            for sid in g.get("expected_source_ids") or []:
                m = re.fullmatch(r"kofia-terms-\d+-(\d+)-(\d+)", sid)
                if m and m.group(1) in version_file:
                    file_name = version_file[m.group(1)]
                    found.add((file_name, article_page(file_name, int(m.group(2)))))
            if found:
                targets[g["case_id"]] = found
    return targets


def is_relevant(doc, target):
    name, page = Path(doc.metadata["source"]).name, doc.metadata["page"]
    return any(name == f and (p is None or page == p) for f, p in target)


def score(docs, target):
    relevant = [is_relevant(d, target) for d in docs[:K]]

    def covered(k):
        return {(f, p) for d in docs[:k] for f, p in target if is_relevant(d, {(f, p)})}

    first = next((i + 1 for i, r in enumerate(relevant) if r), None)
    return {
        **{f"hit@{k}": int(any(relevant[:k])) for k in (1, 3, 5)},
        "precision@5": sum(relevant) / K,
        **{f"recall@{k}": len(covered(k)) / len(target) for k in (3, 5)},
        "mrr": 1 / first if first else 0.0,
    }


def load_chunks(name):
    return [Document(page_content=c["text"], metadata=c, id=c["chunk_id"])
            for c in map(json.loads, (CHUNK_DIR / f"{name}.jsonl").open(encoding="utf-8"))]


def build_searches(name, embeddings):
    vectorstore = Chroma(collection_name=name, embedding_function=embeddings, persist_directory=str(chroma_dir()))
    bm25 = BM25Retriever.from_documents(load_chunks(name), preprocess_func=kiwi_tokenize, k=FETCH_K)
    hybrid = EnsembleRetriever(retrievers=[vectorstore.as_retriever(search_kwargs={"k": FETCH_K}), bm25],
                               weights=WEIGHTS, id_key="chunk_id")
    return {
        "Similarity": lambda q: vectorstore.similarity_search(q, k=K),
        "MMR": lambda q: vectorstore.max_marginal_relevance_search(q, k=K, fetch_k=FETCH_K, lambda_mult=LAMBDA_MULT),
        "BM25": lambda q: bm25.invoke(q)[:K],
        "Hybrid": lambda q: hybrid.invoke(q)[:K],
    }


def table(rows, key):
    # 그룹(난이도 또는 근거 유형)별로 전략마다 한 줄
    cols = ["count", "hit@1", "hit@3", "hit@5", "precision@5", "recall@3", "recall@5", "mrr", "latency_ms"]
    lines = []
    for level in GROUPS[key]:
        selected = [r for r in rows if r[key] == level]
        if not selected:
            continue
        lines.append(f"\n=== {level} ===")
        lines.append(f"{'strategy':>11} |" + "|".join(f"{c:>12} " for c in cols))
        lines.append("-" * 152)
        for s in STRATEGIES:
            rs = [r for r in selected if r["strategy"] == s]
            values = [f"{len(rs):>12} "] + [f"{statistics.mean(r[c] for r in rs):>12.3f} " for c in cols[1:]]
            lines.append(f"{s:>11} |" + "|".join(values))
    return "\n".join(lines)


def evaluate(golden_path, embeddings):
    golden = [json.loads(line) for line in golden_path.open(encoding="utf-8") if line.strip()]
    targets = build_targets(golden)
    skipped = [g["case_id"] for g in golden if g["case_id"] not in targets]
    print(f"\n######## {golden_path.name}: {len(golden)}문항 중 검색 평가 {len(targets)}문항"
          + (f" (정답 문서 없음: {', '.join(skipped)})" if skipped else ""))
    evaluated = [g for g in golden if g["case_id"] in targets]
    print(f"질문 임베딩 {len(evaluated)}개 만드는 중...")
    for g in evaluated:
        embeddings.embed_query(g["question"])

    out_dir = OUT_DIR / golden_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name in COLLECTIONS:
        searches = build_searches(name, embeddings)
        rows = []
        for g in evaluated:
            for s, search in searches.items():
                start = time.perf_counter()
                docs = search(g["question"])
                latency = (time.perf_counter() - start) * 1000
                rows.append({
                    "config": name, "strategy": s, "case_id": g["case_id"],
                    "level": DIFFICULTY[g["difficulty"]], "context_type": g["context_type"],
                    **score(docs, targets[g["case_id"]]), "latency_ms": latency,
                    "retrieved": [f"{Path(d.metadata['source']).name} p{d.metadata['page']}" for d in docs],
                })
        # 같은 검색 결과를 난이도별, 근거 유형별로 각각 묶어 저장한다
        for key, suffix in (("level", ""), ("context_type", "_by_type")):
            report = f"##### {golden_path.stem} / {name} ({'난이도별' if key == 'level' else '근거 유형별'})\n" + table(rows, key)
            print(report)
            (out_dir / f"retrieval_{name}{suffix}.txt").write_text(report + "\n", encoding="utf-8")
        all_rows += rows
    with (out_dir / "retrieval_rows.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(GOLDEN_DIR.glob("*.jsonl"))
    embeddings = QueryCache(get_embeddings())
    for path in paths:
        evaluate(path, embeddings)


if __name__ == "__main__":
    main()
