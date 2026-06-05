# -*- coding: utf-8 -*-
import json
import re

from config import (
    FINAL_RESULT_COUNT,
    PASSAGE_TEXT_STORE_PATH,
    QDRANT_PASSAGE_COLLECTION_NAME,
    QDRANT_VERSES_COLLECTION_NAME,
    VECTOR_CANDIDATE_COUNT,
)
from utils import (
    BIBLE_NAME_MAP,
    jina_embed_text,
    normalize_text,
    qdrant_search_points,
)


PASSAGE_KEYWORDS = [
    "이야기",
    "내용",
    "장면",
    "부분",
    "사건",
    "비유",
    "단락",
    "문맥",
    "설명",
    "방법",
    "어디야",
    "어디에 나와",
    "나오는 곳",
]


def has_passage_keywords(query: str) -> bool:
    q = query.strip()
    if any(k in q for k in PASSAGE_KEYWORDS):
        return True

    # "~한 이야기", "~한 장면" 등은 passage 성격으로 본다.
    return bool(re.search(r".+한\s*(이야기|내용|장면|부분|사건|설명|방법)", q))


def safe_json_array(value):
    try:
        parsed = json.loads(value or "[]")
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


# =========================
# Passage 후보 변환
# =========================

def _to_passage_candidates(hits, passage_text_store):
    candidates = []

    for hit in hits:
        meta = hit.get("payload") or {}
        passage_id = str(meta.get("passage_id") or hit.get("id"))
        score = float(hit.get("score") or 0.0)
        distance = 1.0 - score

        stored = passage_text_store.get(passage_id, {})
        text = stored.get("text", meta.get("text", "본문 없음"))

        candidates.append({
            "id": passage_id,
            "type": "passage",
            "reference": meta.get("reference", ""),
            "book_name": meta.get("book_name", ""),
            "title": normalize_text(meta.get("title", "")),
            "summary": normalize_text(meta.get("summary", "")),
            "distance": distance,
            "score": score,
            "text": text,
        })

    return candidates


def build_passage_candidates(passage_text_store, search_query):
    query_vector = jina_embed_text(search_query, task="retrieval.query")
    hits = qdrant_search_points(
        QDRANT_PASSAGE_COLLECTION_NAME,
        query_vector,
        VECTOR_CANDIDATE_COUNT,
    )
    return _to_passage_candidates(hits, passage_text_store)


# =========================
# Verse 후보 변환
# =========================

def _to_verse_candidates(hits):
    candidates = []

    for hit in hits:
        meta = hit.get("payload") or {}
        score = float(hit.get("score") or 0.0)
        distance = 1.0 - score

        verse_id = str(
            meta.get("verse_id")
            or meta.get("raw_reference")
            or hit.get("id")
        )

        candidates.append({
            "id": verse_id,
            "type": "verse",
            "reference": meta.get("reference", ""),
            "book_name": meta.get("book_name", ""),
            "title": "",
            "summary": "",
            "distance": distance,
            "score": score,
            "text": meta.get("raw") or meta.get("text", "본문 없음"),
            "raw": meta.get("raw", ""),
        })

    return candidates


def build_verse_candidates(search_query, limit=FINAL_RESULT_COUNT):
    query_vector = jina_embed_text(search_query, task="retrieval.query")
    hits = qdrant_search_points(
        QDRANT_VERSES_COLLECTION_NAME,
        query_vector,
        limit,
    )
    return _to_verse_candidates(hits)


# =========================
# 출력
# =========================

def print_result(rank, candidate):
    print("\n" + "=" * 70)
    print(f"{rank}. {candidate['reference']}")

    if candidate.get("type") == "passage":
        print(f"제목: {candidate['title']}")
        print(f"요약: {candidate['summary']}")

    print(f"벡터 거리: {candidate['distance']}")
    print("\n본문:")
    print(candidate["text"])


# =========================
# 직접 레퍼런스 검색
# =========================

def parse_direct_reference(query: str):
    query = query.strip()

    # 예:
    # 창세기 3:16
    # 창 3장 16절
    # 창 3 16
    # 요한일서 1:9
    m = re.match(
        r"^([1-3]?\s*[가-힣]+)\s*(\d+)(?:\s*장\s*|\s*:\s*|\s+)(\d+)(?:\s*절)?$",
        query,
    )
    if m:
        book_part, chap_part, verse_part = m.groups()
        return book_part.replace(" ", ""), int(chap_part), int(verse_part)

    # 예:
    # 창세기 3
    # 창세기 3장
    m = re.match(
        r"^([1-3]?\s*[가-힣]+)\s*(\d+)(?:\s*장)?$",
        query,
    )
    if m:
        book_part, chap_part = m.groups()
        return book_part.replace(" ", ""), int(chap_part), None

    return None, None, None


def get_direct_candidates(query: str, passage_text_store: dict):
    """
    직접 레퍼런스 검색만 담당.
    예:
    - 요한복음 3:16
    - 창 1:1
    - 시편 23장

    본문 텍스트 완전일치 검색은 제거.
    그 역할은 verse vector search가 담당.
    """
    candidates = []

    book_name, target_chap, target_verse = parse_direct_reference(query)

    if not book_name:
        return candidates

    book_abbr = BIBLE_NAME_MAP.get(book_name)

    if not book_abbr:
        return candidates

    for key, val in passage_text_store.items():
        parts = key.split("_")

        if len(parts) != 5:
            continue

        p_book, p_start_chap, p_start_verse, p_end_chap, p_end_verse = parts

        try:
            p_start_chap = int(p_start_chap)
            p_start_verse = int(p_start_verse)
            p_end_chap = int(p_end_chap)
            p_end_verse = int(p_end_verse)
        except ValueError:
            continue

        if p_book != book_abbr:
            continue

        contains = False

        if target_verse is not None:
            # 같은 장 안의 passage
            if p_start_chap == p_end_chap == target_chap:
                contains = p_start_verse <= target_verse <= p_end_verse

            # 여러 장에 걸친 passage
            elif p_start_chap < p_end_chap:
                if target_chap == p_start_chap:
                    contains = target_verse >= p_start_verse
                elif target_chap == p_end_chap:
                    contains = target_verse <= p_end_verse
                elif p_start_chap < target_chap < p_end_chap:
                    contains = True

        else:
            # 장 검색
            contains = p_start_chap <= target_chap <= p_end_chap

        if contains:
            meta = val.get("metadata", {})

            candidates.append({
                "id": key,
                "type": "passage",
                "reference": val.get(
                    "reference",
                    f"{book_abbr} {p_start_chap}:{p_start_verse}-{p_end_chap}:{p_end_verse}",
                ),
                "book_name": val.get("book_name", meta.get("book_name", "")),
                "title": normalize_text(meta.get("title", "")),
                "summary": normalize_text(meta.get("summary", "")),
                "distance": 0.0,
                "score": 1.0,
                "text": val.get("text", "본문 없음"),
                "is_direct_match": True,
            })

    return candidates


# =========================
# 검색 라우팅
# =========================

def should_search_verse(query: str):
    q = query.strip()

    # 직접 주소 검색은 get_direct_candidates에서 처리하므로 여기서는 verse vector로 안 보냄
    book_name, target_chap, target_verse = parse_direct_reference(q)
    if book_name:
        return False

    # 짧은 구문은 verse 검색
    if len(q) <= 30:
        return True

    # 따옴표로 감싼 문장도 verse 검색으로 간주
    if q.startswith('"') or q.startswith("'") or q.startswith("“") or q.startswith("‘"):
        return True

    return False


def search_auto(query: str, passage_text_store: dict):
    # 1. 레퍼런스 직접 검색
    direct_candidates = get_direct_candidates(query, passage_text_store)

    if direct_candidates:
        return "direct", direct_candidates[:FINAL_RESULT_COUNT]

    # 2. 절 벡터 검색이 필요한지 판단
    # passage 키워드가 있어도 verse 80% 이상이면 verse로 채택한다.
    if should_search_verse(query) or has_passage_keywords(query):
        verse_candidates = build_verse_candidates(
            search_query=query,
            limit=FINAL_RESULT_COUNT,
        )

        verse_distance_threshold = 0.60
        if has_passage_keywords(query):
            verse_distance_threshold = 0.20  # 80% 이상 일치

        if any(c.get("distance", 1.0) <= verse_distance_threshold for c in verse_candidates):
            return "verse", verse_candidates[:FINAL_RESULT_COUNT]
        else:
            # 그렇지 않으면 passage 검색으로 간주
            pass
    # 3. 이야기/내용 → passage 벡터 검색
    passage_candidates = build_passage_candidates(
        passage_text_store=passage_text_store,
        search_query=query,
    )
    return "passage", passage_candidates[:FINAL_RESULT_COUNT]


# =========================
# 메인
# =========================

def main():
    print("Jina + Qdrant 검색 클라이언트 준비 중...")

    with open(PASSAGE_TEXT_STORE_PATH, "r", encoding="utf-8") as f:
        passage_text_store = json.load(f)

    while True:
        query = input("\n검색어를 입력하세요. 종료하려면 q: ").strip()

        if query.lower() in ["q", "quit", "exit"]:
            break

        if not query:
            continue

        mode, final_results = search_auto(query, passage_text_store)

        if not final_results:
            print("검색 결과가 없습니다.")
            continue

        print(f"\n검색 모드: {mode}")

        for rank, candidate in enumerate(final_results, start=1):
            print_result(rank, candidate)


if __name__ == "__main__":
    main()