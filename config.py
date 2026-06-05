import os
from dotenv import load_dotenv
load_dotenv()

def _getenv_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return int(value)
	except ValueError:
		return default


def apply_runtime_env() -> None:
	os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "4"))
	os.environ.setdefault("MKL_NUM_THREADS", os.getenv("MKL_NUM_THREADS", "4"))
	os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))


apply_runtime_env()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 일부 기존 코드/문서 호환을 위해 data/와 루트 파일 위치를 모두 지원한다.
BIBLE_TXT_PATH = os.getenv("BIBLE_TXT_PATH", os.path.join(BASE_DIR, "bible_ko.txt"))
if not os.path.exists(BIBLE_TXT_PATH):
	BIBLE_TXT_PATH = os.path.join(DATA_DIR, "bible_ko.txt")

PASSAGE_TEXT_STORE_PATH = os.getenv(
	"PASSAGE_TEXT_STORE_PATH",
	os.path.join(DATA_DIR, "passage_texts.json"),
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_BASE_URL = QDRANT_URL.rstrip("/")
QDRANT_HEADERS = {"Content-Type": "application/json"}
if QDRANT_API_KEY:
	QDRANT_HEADERS["api-key"] = QDRANT_API_KEY

QDRANT_PASSAGE_COLLECTION_NAME = os.getenv(
	"QDRANT_PASSAGE_COLLECTION_NAME",
	os.getenv("QDRANT_COLLECTION_NAME", "bible_passage_collection"),
)
QDRANT_VERSES_COLLECTION_NAME = os.getenv("QDRANT_VERSES_COLLECTION_NAME", "bible_verses_collection")

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBED_MODEL = os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v3-text")
JINA_EMBED_DIM = _getenv_int("JINA_EMBED_DIM", 1024)
JINA_EMBED_API_URL = os.getenv("JINA_EMBED_API_URL", "https://api.jina.ai/v1/embeddings")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "jina").strip().lower()
LOCAL_EMBED_MODEL = os.getenv(
	"LOCAL_EMBED_MODEL",
	"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
LOCAL_EMBED_DEVICE = os.getenv("LOCAL_EMBED_DEVICE", "cpu")
LOCAL_EMBED_BATCH_SIZE = _getenv_int("LOCAL_EMBED_BATCH_SIZE", 64)
LOCAL_EMBED_NORMALIZE = os.getenv("LOCAL_EMBED_NORMALIZE", "false").strip().lower() in (
	"1",
	"true",
	"yes",
	"on",
)

BATCH_SIZE = _getenv_int("BATCH_SIZE", 40)
JINA_MAX_RETRIES = _getenv_int("JINA_MAX_RETRIES", 10)
JINA_RETRY_BASE_DELAY = float(os.getenv("JINA_RETRY_BASE_DELAY", "2.0"))
JINA_RETRY_MAX_DELAY = float(os.getenv("JINA_RETRY_MAX_DELAY", "90.0"))

VECTOR_CANDIDATE_COUNT = _getenv_int("VECTOR_CANDIDATE_COUNT", 50)
FINAL_RESULT_COUNT = _getenv_int("FINAL_RESULT_COUNT", 5)

STATIC_PAGE_API_KEY = os.getenv("STATIC_PAGE_API_KEY", "fc5b4866e43b0390d8f90280191dc480")
SEARCH_QUERY_MAX_LEN = _getenv_int("SEARCH_QUERY_MAX_LEN", 200)
SEARCH_RATE_LIMIT_PER_MIN = _getenv_int("SEARCH_RATE_LIMIT_PER_MIN", 10)
SEARCH_CACHE_TTL_SEC = _getenv_int("SEARCH_CACHE_TTL_SEC", 180)
SEARCH_CACHE_MAX_ITEMS = _getenv_int("SEARCH_CACHE_MAX_ITEMS", 500)
