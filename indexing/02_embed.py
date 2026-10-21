"""06_chunks/chunks_*.jsonl → 임베딩 → Chroma 벡터 DB (설정마다 컬렉션 하나)

- 임베딩 모델: gemini-embedding-2 (강의와 같음)
- 컬렉션 이름: chunks_300_50, chunks_500_60
- 청크 id를 Chroma id로 써서, 다시 돌리면 이미 넣은 청크는 건너뛴다 (새 청크만 임베딩 → 비용 절약).
- 청킹을 바꿔서 글이 달라진 청크가 있으면 컬렉션을 새로 만든다. 글이 같은 청크는 옛 벡터를 그대로 옮기고, 달라진 것만 임베딩한다.
- 벡터 DB 위치는 common.chroma_dir() (한글 경로면 영문 경로로 자동 변경). 커밋하지 않는다.
"""
import json
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

from common import BASE, EMBEDDING_MODEL_NAME, chroma_dir, get_embeddings, log_cost

CHUNK_DIR = BASE / "06_chunks"
BATCH = 100


def load_chunks(path):
    docs = []
    for line in path.open(encoding="utf-8"):
        c = json.loads(line)
        meta = {k: c[k] for k in ("chunk_id", "source", "page", "doc_group", "block_type")}
        docs.append(Document(page_content=c["text"], metadata=meta, id=c["chunk_id"]))
    return docs


def main():
    persist_dir = chroma_dir()
    embeddings = get_embeddings()
    for path in sorted(CHUNK_DIR.glob("chunks_*.jsonl")):
        name = path.stem  # chunks_300_50
        docs = load_chunks(path)
        # Chroma는 HNSW 색인을 sync_threshold(기본 1000)개마다만 디스크에 쓴다. 나머지는 로그에 남았다가 열 때 다시 채우는데,
        # 이 상태에서 검색 결과가 같은 질문에도 바뀌었다(회고 참고). 청크 수로 맞춰 다 넣은 순간 한 번에 저장되게 한다.
        # ponytail: 컬렉션 만들 때만 적용됨, 중간에 끊긴 뒤 이어 넣으면 다시 안 맞을 수 있음 → chroma_db 지우고 처음부터
        def open_store():
            return Chroma(collection_name=name, embedding_function=embeddings, persist_directory=str(persist_dir),
                          collection_metadata={"hnsw:batch_size": len(docs), "hnsw:sync_threshold": len(docs)})

        store = open_store()
        old = store.get(include=["documents", "embeddings"])
        old_vectors = dict(zip(old["documents"], old["embeddings"]))  # 글 → 벡터
        changed = old["ids"] and (set(old["ids"]) != {d.id for d in docs} or any(d.page_content not in old_vectors for d in docs))
        if changed:
            # 글이 바뀐 청크가 있다 → 컬렉션을 새로 만들고, 글이 같은 청크는 옛 벡터를 옮긴다 (임베딩 비용 없음)
            store.delete_collection()
            store = open_store()
            reuse = [d for d in docs if d.page_content in old_vectors]
            for i in range(0, len(reuse), 1000):
                batch = reuse[i:i + 1000]
                store._collection.add(ids=[d.id for d in batch], documents=[d.page_content for d in batch],
                                      embeddings=[old_vectors[d.page_content] for d in batch],
                                      metadatas=[d.metadata for d in batch])
            print(f"{name}: 청킹이 바뀌어 컬렉션을 새로 만듦, 옛 벡터 재사용 {len(reuse)}개")
        existing = set(store.get(include=[])["ids"])
        todo = [d for d in docs if d.id not in existing]
        print(f"{name}: 청크 {len(docs)}개 중 새로 임베딩 {len(todo)}개")
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            for attempt in range(5):  # 속도 제한(429) 등에 걸리면 기다렸다 다시
                try:
                    store.add_documents(batch, ids=[d.id for d in batch])
                    break
                except Exception as e:
                    wait = 10 * (attempt + 1)
                    print(f"  재시도 {attempt + 1} ({wait}초 후): {str(e)[:120]}")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"{name} {i}번째 묶음 실패")
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}")
        if todo:
            log_cost(f"청크 임베딩 ({name})", EMBEDDING_MODEL_NAME, len(todo),
                     round(sum(len(d.page_content) for d in todo) * 0.66), note="추정 (글자 × 0.66)")
        print(f"{name}: 저장 완료 → 컬렉션 {store._collection.count()}개")


if __name__ == "__main__":
    main()
