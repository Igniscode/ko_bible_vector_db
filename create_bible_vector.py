# -*- coding: utf-8 -*-
import os
import json
import uuid
import time
import random

from config import (
    BATCH_SIZE,
    JINA_EMBED_DIM,
    PASSAGE_TEXT_STORE_PATH,
    QDRANT_PASSAGE_COLLECTION_NAME,
)
from utils import jina_embed_texts, qdrant_ensure_collection, qdrant_upsert_points

def normalize_metadata(metadata):
    if not isinstance(metadata, dict):
        metadata = {}

    title = str(metadata.get("title", "") or "").strip()[:40]
    summary = str(metadata.get("summary", "") or "").strip()[:200]

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


def flush_batch(batch_ids, batch_documents, batch_metadatas):
    if not batch_ids:
        return 0

    vectors = jina_embed_texts(batch_documents, task="retrieval.passage")
    points = []
    for point_id, vector, payload in zip(batch_ids, vectors, batch_metadatas):
        points.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"bible-passage:{point_id}")),
                "vector": vector,
                "payload": payload,
            }
        )

    qdrant_upsert_points(QDRANT_PASSAGE_COLLECTION_NAME, points, wait=False)

    count = len(batch_ids)
    batch_ids.clear()
    batch_documents.clear()
    batch_metadatas.clear()
    return count


# =========================
# 메인
# =========================

def main():
    print("Qdrant 준비")
    qdrant_ensure_collection(QDRANT_PASSAGE_COLLECTION_NAME, JINA_EMBED_DIM)

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
                processed_count += flush_batch(batch_ids, batch_documents, batch_metadatas)
                print(f"[{idx}/{total}] batch 저장 완료 | 처리 {processed_count}, 에러 {error_count}")

        except Exception as e:
            error_count += 1
            print(f"[{idx}/{total}] {passage_id} 에러: {e}")

    processed_count += flush_batch(batch_ids, batch_documents, batch_metadatas)

    print("\n완료")
    print(f"처리 대상: {total}")
    print(f"Qdrant upsert 처리: {processed_count}")
    print(f"에러: {error_count}")
    print(f"입력 파일: {PASSAGE_TEXT_STORE_PATH}")
    print(f"Collection: {QDRANT_PASSAGE_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
