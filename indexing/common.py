"""indexing 스크립트들이 같이 쓰는 설정: 벡터 DB 위치, 임베딩 모델."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

BASE = Path(__file__).parent.parent  # 프로젝트 루트
EMBEDDING_MODEL_NAME = "gemini-embedding-2"
COLLECTIONS = ["chunks_300_50", "chunks_500_60"]

load_dotenv(override=True)
os.environ["LANGSMITH_TRACING"] = "false"


def chroma_dir():
    # Chroma는 Windows에서 경로에 한글이 있으면 HNSW 색인 파일(.bin)을 오류 없이 안 써서,
    # 다시 열 때 "Error loading hnsw index"가 난다. 그래서 한글 경로면 영문 경로에 저장한다.
    # CHROMA_DIR 환경변수로 직접 지정할 수도 있다.
    if os.environ.get("CHROMA_DIR"):
        return Path(os.environ["CHROMA_DIR"])
    default = BASE / "chroma_db"
    if str(default).isascii() or os.name != "nt":
        return default
    fallback = Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "financial-rag-agent" / "chroma_db"
    print(f"[알림] 프로젝트 경로에 한글이 있어 벡터 DB를 영문 경로에 저장합니다: {fallback}")
    return fallback


def get_embeddings():
    # .env에 GOOGLE_API_KEY(만료)와 GEMINI_API_KEY가 같이 있으면 GOOGLE_API_KEY를 먼저 써서 Gemini 키를 직접 넘긴다.
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=os.environ["GEMINI_API_KEY"])
