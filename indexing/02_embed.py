"""06_chunks/chunks_*.jsonl → 임베딩 → Chroma 벡터 DB (설정마다 컬렉션 하나)

- 임베딩 모델: gemini-embedding-2 (강의와 같음)
- 컬렉션 이름: chunks_300_50, chunks_500_60
- 청크 id를 Chroma id로 써서, 다시 돌리면 이미 넣은 청크는 건너뛴다 (새 청크만 임베딩 → 비용 절약).
- 벡터 DB 위치는 common.chroma_dir() (한글 경로면 영문 경로로 자동 변경). 커밋하지 않는다.
"""
import json
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

from common import BASE, chroma_dir, get_embeddings

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
        store = Chroma(collection_name=name, embedding_function=embeddings, persist_directory=str(persist_dir),
                       collection_metadata={"hnsw:batch_size": len(docs), "hnsw:sync_threshold": len(docs)})
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
        print(f"{name}: 저장 완료 → 컬렉션 {store._collection.count()}개")


if __name__ == "__main__":
    main()
