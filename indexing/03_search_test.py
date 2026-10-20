"""두 컬렉션(300/50, 500/60)에 같은 질문을 던져 상위 결과를 나란히 본다 (동작 확인용).

python indexing/03_search_test.py "질문"   # 질문을 직접 넣을 수도 있다
"""
import sys

from langchain_chroma import Chroma

from common import COLLECTIONS, chroma_dir, get_embeddings

QUESTIONS = sys.argv[1:] or [
    "휴면성 증권계좌는 어떤 계좌인가요?",                      # 약관·용어 (본문)
    "신탁방식과 저당권방식 주택연금은 배우자 연금승계가 어떻게 다른가요?",  # 비교표 (표)
    "상각채권 기본감면율은 어떻게 계산하나요?",                 # 수식
    "연령별 금융채무불이행자 수는 몇 명인가요?",                # 그래프
]

embeddings = get_embeddings()
persist_dir = str(chroma_dir())
stores = {name: Chroma(collection_name=name, embedding_function=embeddings, persist_directory=persist_dir)
          for name in COLLECTIONS}

for q in QUESTIONS:
    print(f"\n### {q}")
    for name, store in stores.items():
        print(f"  [{name}]")
        for doc, score in store.similarity_search_with_score(q, k=3):
            m = doc.metadata
            preview = " ".join(doc.page_content.split())[:70]
            print(f"    {score:.3f} {m['source'].split('/')[-1]} {m['page']}쪽 ({m['block_type']}) {preview}")
