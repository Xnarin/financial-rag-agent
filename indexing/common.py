"""indexing·evaluation 스크립트들이 같이 쓰는 설정: 벡터 DB 위치, 임베딩 모델, 비용 기록."""
import datetime
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


COST_LOG = BASE / "docs" / "비용기록.md"
# 1M 토큰당 달러 (입력, 출력). 출력에는 생각(thinking) 토큰도 포함. 2026-12-31까지 단가
PRICES = {"gemini-3.6-flash": (0.75, 3.75), "gemini-3.5-flash-lite": (0.30, 2.50), EMBEDDING_MODEL_NAME: (0.20, 0.0)}


def log_cost(task, model, calls, input_tokens, output_tokens=0, note=""):
    # 유료 호출을 할 때마다 docs/비용기록.md 표에 한 줄 추가하고 달러 금액을 돌려준다
    price_in, price_out = PRICES[model]
    usd = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    with COST_LOG.open("a", encoding="utf-8") as f:
        f.write(f"| {datetime.date.today()} | {task} | `{model}` | {calls:,} | {input_tokens:,} | {output_tokens:,} | ${usd:.4f} | {note} |\n")
    print(f"[비용] {task}: {calls}회, 입력 {input_tokens:,} / 출력 {output_tokens:,} 토큰 → ${usd:.4f} (docs/비용기록.md)")
    return usd


def get_embeddings():
    # .env에 GOOGLE_API_KEY(만료)와 GEMINI_API_KEY가 같이 있으면 GOOGLE_API_KEY를 먼저 써서 Gemini 키를 직접 넘긴다.
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=os.environ["GEMINI_API_KEY"])
