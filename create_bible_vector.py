# -*- coding: utf-8 -*-
import os
import json
import uuid
import time
import random

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests


# =========================
# 설정
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PASSAGE_TEXT_STORE_PATH = os.path.join(DATA_DIR, "passage_texts.json")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "bible_passage_collection")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBED_MODEL = os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v3-text")
JINA_EMBED_DIM = int(os.getenv("JINA_EMBED_DIM", "1024"))
JINA_EMBED_API_URL = os.getenv("JINA_EMBED_API_URL", "https://api.jina.ai/v1/embeddings")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "40"))
JINA_MAX_RETRIES = int(os.getenv("JINA_MAX_RETRIES", "10"))
JINA_RETRY_BASE_DELAY = float(os.getenv("JINA_RETRY_BASE_DELAY", "2.0"))
JINA_RETRY_MAX_DELAY = float(os.getenv("JINA_RETRY_MAX_DELAY", "90.0"))


# =========================
# 유틸
# =========================

def ensure_project_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def _qdrant_headers():
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    return headers


def _qdrant_base_url():
    return QDRANT_URL.rstrip("/")


def create_qdrant_client():
    base_url = _qdrant_base_url()
    headers = _qdrant_headers()

    get_resp = requests.get(
        f"{base_url}/collections/{COLLECTION_NAME}",
        headers=headers,
        timeout=60,
    )

    if get_resp.status_code == 404:
        create_resp = requests.put(
            f"{base_url}/collections/{COLLECTION_NAME}",
            headers=headers,
            json={
                "vectors": {
                    "size": JINA_EMBED_DIM,
                    "distance": "Cosine",
                }
            },
            timeout=60,
        )
        create_resp.raise_for_status()
    elif get_resp.status_code >= 400:
        get_resp.raise_for_status()

    return {
        "base_url": base_url,
        "headers": headers,
    }


def jina_embed_texts(texts, task="retrieval.passage"):
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY가 설정되어 있지 않습니다.")

    if not texts:
        return []

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": JINA_EMBED_MODEL,
        "input": texts,
        "task": task,
        "dimensions": JINA_EMBED_DIM,
    }

    for attempt in range(JINA_MAX_RETRIES + 1):
        resp = None
        try:
            resp = requests.post(
                JINA_EMBED_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if resp.status_code == 429:
                if attempt >= JINA_MAX_RETRIES:
                    resp.raise_for_status()

                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = min(JINA_RETRY_MAX_DELAY, JINA_RETRY_BASE_DELAY * (2 ** attempt))
                else:
                    delay = min(JINA_RETRY_MAX_DELAY, JINA_RETRY_BASE_DELAY * (2 ** attempt))

                # 다중 클라이언트 충돌을 줄이기 위해 랜덤 지터를 추가한다.
                delay = delay + random.uniform(0, max(0.3, delay * 0.2))
                print(f"[Jina 429] {attempt + 1}회 재시도 대기: {delay:.1f}초")
                time.sleep(delay)
                continue

            if 500 <= resp.status_code < 600:
                if attempt >= JINA_MAX_RETRIES:
                    resp.raise_for_status()

                delay = min(JINA_RETRY_MAX_DELAY, JINA_RETRY_BASE_DELAY * (2 ** attempt))
                delay = delay + random.uniform(0, max(0.3, delay * 0.2))
                print(f"[Jina {resp.status_code}] {attempt + 1}회 재시도 대기: {delay:.1f}초")
                time.sleep(delay)
                continue

            resp.raise_for_status()
            break

        except requests.RequestException:
            if attempt >= JINA_MAX_RETRIES:
                raise

            delay = min(JINA_RETRY_MAX_DELAY, JINA_RETRY_BASE_DELAY * (2 ** attempt))
            delay = delay + random.uniform(0, max(0.3, delay * 0.2))
            print(f"[Jina 네트워크 예외] {attempt + 1}회 재시도 대기: {delay:.1f}초")
            time.sleep(delay)
            continue

    data = resp.json().get("data", [])
    if len(data) != len(texts):
        raise RuntimeError("Jina embedding 결과 개수가 입력 개수와 다릅니다.")

    data.sort(key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in data]


def clean_item(text, max_len):
    text = str(text or "").strip()
    return text[:max_len]


def normalize_metadata(metadata):
    if not isinstance(metadata, dict):
        metadata = {}

    title = clean_item(metadata.get("title", ""), 40)
    summary = clean_item(metadata.get("summary", ""), 200)

    return {
        "title": title,
        "summary": summary,
    }


def make_embedding_document(reference, metadata):
    return f"""
{reference}
{metadata['title']}
{metadata['summary']}
""".strip()


def make_vector_metadata(item, metadata):
    passage_id = str(item.get("id") or "")
    return {
        "passage_id": passage_id,
        "reference": item.get("reference", ""),
        "book_abbr": item.get("book_abbr", ""),
        "book_name": item.get("book_name", ""),
        "start_chapter": int(item.get("start_chapter", 0)),
        "start_verse": int(item.get("start_verse", 0)),
        "end_chapter": int(item.get("end_chapter", 0)),
        "end_verse": int(item.get("end_verse", 0)),
        "title": metadata["title"],
        "summary": metadata["summary"],
    }


def to_qdrant_point_id(passage_id: str) -> str:
    # Qdrant point id must be uint64 or UUID.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"bible-passage:{passage_id}"))


def load_passage_items():
    if not os.path.exists(PASSAGE_TEXT_STORE_PATH):
        raise FileNotFoundError(f"passage_texts.json 파일이 없습니다: {PASSAGE_TEXT_STORE_PATH}")

    with open(PASSAGE_TEXT_STORE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # 기존 구조: { passage_id: { ... } }
        return [(str(k), v if isinstance(v, dict) else {}) for k, v in data.items()]

    if isinstance(data, list):
        # 혹시 리스트 구조일 경우도 처리
        rows = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or i)
            rows.append((pid, item))
        return rows

    raise ValueError("passage_texts.json 형식이 dict 또는 list가 아닙니다.")


def flush_batch(collection, batch_ids, batch_documents, batch_metadatas):
    if not batch_ids:
        return 0

    vectors = jina_embed_texts(batch_documents, task="retrieval.passage")
    points = []
    for point_id, vector, payload in zip(batch_ids, vectors, batch_metadatas):
        points.append(
            {
                "id": to_qdrant_point_id(str(point_id)),
                "vector": vector,
                "payload": payload,
            }
        )

    upsert_resp = requests.put(
        f"{collection['base_url']}/collections/{COLLECTION_NAME}/points",
        headers=collection["headers"],
        params={"wait": "false"},
        json={"points": points},
        timeout=120,
    )
    upsert_resp.raise_for_status()

    count = len(batch_ids)
    batch_ids.clear()
    batch_documents.clear()
    batch_metadatas.clear()
    return count


# =========================
# 메인
# =========================

def main():
    ensure_project_dirs()

    print("Qdrant 준비")
    collection = create_qdrant_client()

    print("passage_texts.json 로딩")
    rows = load_passage_items()
    total = len(rows)
    print(f"총 {total}개 passage를 읽었습니다.")

    batch_ids = []
    batch_documents = []
    batch_metadatas = []

    processed_count = 0
    error_count = 0

    for idx, (passage_id, item) in enumerate(rows, start=1):
        try:
            item = dict(item)
            item["id"] = str(passage_id)

            reference = item.get("reference", passage_id)
            metadata = normalize_metadata(item.get("metadata", {}))

            document_text = make_embedding_document(reference, metadata)
            vector_metadata = make_vector_metadata(item, metadata)

            batch_ids.append(passage_id)
            batch_documents.append(document_text)
            batch_metadatas.append(vector_metadata)

            if len(batch_ids) >= BATCH_SIZE:
                processed_count += flush_batch(
                    collection,
                    batch_ids,
                    batch_documents,
                    batch_metadatas,
                )
                print(f"[{idx}/{total}] batch 저장 완료 | 처리 {processed_count}, 에러 {error_count}")

        except Exception as e:
            error_count += 1
            print(f"[{idx}/{total}] {passage_id} 에러: {e}")

    processed_count += flush_batch(
        collection,
        batch_ids,
        batch_documents,
        batch_metadatas,
    )

    print("\n완료")
    print(f"처리 대상: {total}")
    print(f"Qdrant upsert 처리: {processed_count}")
    print(f"에러: {error_count}")
    print(f"입력 파일: {PASSAGE_TEXT_STORE_PATH}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
