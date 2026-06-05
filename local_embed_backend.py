# -*- coding: utf-8 -*-
from config import (
    LOCAL_EMBED_BATCH_SIZE,
    LOCAL_EMBED_DEVICE,
    LOCAL_EMBED_MODEL,
    LOCAL_EMBED_NORMALIZE,
)


_local_embedder = None


def _normalize_task(task):
    if not task:
        return None

    # Jina API task labels are fine-grained (e.g. retrieval.query/passage),
    # but some local HF models only accept coarse task names.
    if task.startswith("retrieval"):
        return "retrieval"

    return task


def _get_local_embedder():
    global _local_embedder

    if _local_embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "로컬 임베딩 사용 시 sentence-transformers 설치가 필요합니다. "
                "pip install -r requirements-local.txt"
            ) from e

        _local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL, device=LOCAL_EMBED_DEVICE, trust_remote_code=True)

    return _local_embedder


def embed_texts(texts, task=None):
    if not texts:
        return []

    model = _get_local_embedder()
    normalized_task = _normalize_task(task)
    vectors = model.encode(
        texts,
        task=normalized_task,
        batch_size=LOCAL_EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=LOCAL_EMBED_NORMALIZE,
    )
    return vectors.tolist()
