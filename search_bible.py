# -*- coding: utf-8 -*-
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import requests
from dotenv import load_dotenv


# =========================
# 설정
# =========================

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "bible_passage_collection")
PASSAGE_TEXT_STORE_PATH = "data/passage_texts.json"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBED_MODEL = os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v3-text")
JINA_EMBED_DIM = int(os.getenv("JINA_EMBED_DIM", "1024"))
JINA_EMBED_API_URL = os.getenv("JINA_EMBED_API_URL", "https://api.jina.ai/v1/embeddings")

VECTOR_CANDIDATE_COUNT = 50
FINAL_RESULT_COUNT = 5


def get_qdrant_client():
    # app.py와의 기존 인터페이스 유지를 위해 연결 설정만 반환한다.
    return {
        "base_url": QDRANT_URL.rstrip("/"),
        "api_key": QDRANT_API_KEY,
    }


def _qdrant_headers():
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    return headers


def _qdrant_search_points(query_vector, limit):
    endpoint = f"{QDRANT_URL.rstrip('/')}" + f"/collections/{COLLECTION_NAME}/points/query"
    payload = {
        "query": query_vector,
        "limit": limit,
        "with_payload": True,
    }

    resp = requests.post(
        endpoint,
        headers=_qdrant_headers(),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()

    body = resp.json()
    points = body.get("result", {}).get("points", [])
    if isinstance(points, list):
        return points
    return []


def jina_embed_text(text: str, task: str):
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY가 설정되어 있지 않습니다.")

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": JINA_EMBED_MODEL,
        "input": [text],
        "task": task,
        "dimensions": JINA_EMBED_DIM,
    }

    resp = requests.post(
        JINA_EMBED_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()

    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError("Jina embedding 응답이 비어 있습니다.")

    return data[0]["embedding"]


# =========================
# 유틸
# =========================

def safe_json_array(value):
    try:
        parsed = json.loads(value or "[]")
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _to_candidates(hits, passage_text_store):
    candidates = []

    for hit in hits:
        meta = hit.get("payload") or {}
        passage_id = str(meta.get("passage_id") or hit.get("id"))
        score = float(hit.get("score") or 0.0)
        distance = 1.0 - score

        stored = passage_text_store.get(passage_id, {})
        text = stored.get("text", "본문 없음")

        tags = safe_json_array(meta.get("tags_json", "[]"))
        modern_queries = safe_json_array(meta.get("modern_queries_json", "[]"))

        candidates.append({
            "id": passage_id,
            "reference": meta.get("reference", ""),
            "book_name": meta.get("book_name", ""),
            "title": normalize_text(meta.get("title", "")),
            "summary": normalize_text(meta.get("summary", "")),
            "tags": [normalize_text(x) for x in tags],
            "modern_queries": [normalize_text(x) for x in modern_queries],
            "distance": distance,
            "text": text,
        })

    return candidates


def build_hybrid_candidates(collection, passage_text_store, search_query):
    # 휴리스틱 보강 조회 없이 단일 벡터 검색만 수행한다.
    _ = collection
    query_vector = jina_embed_text(search_query, task="retrieval.query")
    hits = _qdrant_search_points(query_vector, VECTOR_CANDIDATE_COUNT)
    return _to_candidates(hits, passage_text_store)


# =========================
# 출력
# =========================

def print_result(rank, candidate):
    print("\n" + "=" * 70)
    print(f"{rank}. {candidate['reference']}")
    print(f"제목: {candidate['title']}")
    print(f"요약: {candidate['summary']}")
    print(f"태그: {', '.join(candidate['tags'])}")
    print(f"벡터 거리: {candidate['distance']}")

    if candidate["modern_queries"]:
        print("\n현대어 연결:")
        for q in candidate["modern_queries"]:
            print(f"- {q}")

    print("\n본문:")
    print(candidate["text"])


# =========================
# 구절 직접 검색 지원
# =========================

BIBLE_NAME_MAP = {
    # 구약
    "창세기": "창", "창": "창",
    "출애굽기": "출", "출": "출",
    "레위기": "레", "레": "레",
    "민수기": "민", "민": "민",
    "신명기": "신", "신": "신",
    "여호수아": "수", "수": "수",
    "사사기": "삿", "삿": "삿",
    "룻기": "룻", "룻": "룻",
    "사무엘상": "삼상", "삼상": "삼상", "사무엘전": "삼상",
    "사무엘하": "삼하", "삼하": "삼하", "사무엘후": "삼하",
    "열왕기상": "왕상", "왕상": "왕상", "열왕기전": "왕상",
    "열왕기하": "왕하", "왕하": "왕하", "열왕기후": "왕하",
    "역대기상": "대상", "대상": "대상", "역대전": "대상", "역대지략상": "대상",
    "역대기하": "대하", "대하": "대하", "역대후": "대하", "역대지략하": "대하",
    "에스라": "스", "스": "스",
    "느헤미야": "느", "느": "느",
    "에스더": "에", "에": "에",
    "욥기": "욥", "욥": "욥",
    "시편": "시", "시": "시",
    "잠언": "잠", "잠": "잠",
    "전도서": "전", "전": "전",
    "아가": "아", "아": "아",
    "이사야": "사", "사": "사",
    "예레미야": "렘", "렘": "렘",
    "예레미야애가": "애", "예레미야 에가": "애", "애가": "애", "애": "애",
    "에스겔": "겔", "겔": "겔",
    "다니엘": "단", "단": "단",
    "호세아": "호", "호": "호",
    "요엘": "욜", "욜": "욜",
    "아모스": "암", "암": "암",
    "오바디야": "옵", "오바디아": "옵", "옵": "옵",
    "요나": "욘", "욘": "욘",
    "미가": "미", "미": "미",
    "나훔": "나", "나": "나",
    "하박국": "합", "합": "합",
    "스바냐": "습", "습": "습",
    "학개": "학", "학": "학",
    "스가랴": "슥", "슥": "슥",
    "말라기": "말", "말": "말",
    
    # 신약
    "마태복음": "마", "마태": "마", "마": "마",
    "마가복음": "막", "마가": "막", "막": "막",
    "누가복음": "눅", "누가": "눅", "눅": "눅",
    "요한복음": "요", "요한": "요", "요": "요",
    "사도행전": "행", "사도": "행", "행": "행",
    "로마서": "롬", "로마": "롬", "롬": "롬",
    "고린도전서": "고전", "고전": "고전",
    "고린도후서": "고후", "고후": "고후",
    "갈라디아서": "갈", "갈라디아": "갈", "갈": "갈",
    "에베소서": "엡", "에베소": "엡", "엡": "엡",
    "빌립보서": "빌", "빌립보": "빌", "빌": "빌",
    "골로새서": "골", "골로새": "골", "골": "골",
    "데살로니가전서": "살전", "살전": "살전",
    "데살로니가후서": "살후", "살후": "살후",
    "디모데전서": "딤전", "딤전": "딤전",
    "디모데후서": "딤후", "딤후": "딤후",
    "디도서": "딛", "디도": "딛", "딛": "딛",
    "빌레몬서": "몬", "빌레몬": "몬", "몬": "몬",
    "히브리서": "히", "히브리": "히", "히": "히",
    "야고보서": "약", "야고보": "약", "약": "약",
    "베드로전서": "벧전", "벧전": "벧전",
    "베드로후서": "벧후", "벧후": "벧후",
    "요한일서": "요일", "요일": "요일", "요한1서": "요일",
    "요한이서": "요이", "요이": "요이", "요한2서": "요이",
    "요한삼서": "요삼", "요삼": "요삼", "요한3서": "요삼",
    "유다서": "유", "유": "유",
    "요한계시록": "계", "계시록": "계", "계": "계"
}

def parse_direct_reference(query: str):
    query = query.strip()
    
    # 장/절 패턴 (예: "창세기 3:16", "창 3장 16절", "창 3 16", "요한일서 1:9")
    m = re.match(r"^([1-3]?\s*[가-힣]+)\s*(\d+)(?:\s*장\s*|\s*:\s*|\s+)(\d+)(?:\s*절)?$", query)
    if m:
        book_part, chap_part, verse_part = m.groups()
        return book_part.replace(" ", ""), int(chap_part), int(verse_part)
        
    # 장 패턴 (예: "창세기 3", "창세기 3장")
    m = re.match(r"^([1-3]?\s*[가-힣]+)\s*(\d+)(?:\s*장)?$", query)
    if m:
        book_part, chap_part = m.groups()
        return book_part.replace(" ", ""), int(chap_part), None
        
    return None, None, None

def get_direct_candidates(query: str, passage_text_store: dict):
    candidates = []
    
    # 1) 성경 구절 레퍼런스 검색 (예: "요한복음 3:16", "창 3장")
    book_name, target_chap, target_verse = parse_direct_reference(query)
    if book_name:
        book_abbr = BIBLE_NAME_MAP.get(book_name)
        if book_abbr:
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
                    if p_start_chap == p_end_chap == target_chap:
                        contains = (p_start_verse <= target_verse <= p_end_verse)
                    elif p_start_chap < p_end_chap:
                        if target_chap == p_start_chap:
                            contains = (target_verse >= p_start_verse)
                        elif target_chap == p_end_chap:
                            contains = (target_verse <= p_end_verse)
                        elif p_start_chap < target_chap < p_end_chap:
                            contains = True
                else:
                    contains = (p_start_chap <= target_chap <= p_end_chap)
                
                if contains:
                    meta = val.get("metadata", {})
                    candidates.append({
                        "id": key,
                        "reference": val.get("reference", f"{book_abbr} {p_start_chap}:{p_start_verse}-{p_end_chap}:{p_end_verse}"),
                        "book_name": meta.get("book_name", ""),
                        "title": meta.get("title", ""),
                        "summary": meta.get("summary", ""),
                        "tags": meta.get("tags", []),
                        "modern_queries": meta.get("modern_queries", []),
                        "distance": 0.0,
                        "text": val.get("text", "본문 없음"),
                        "is_direct_match": True
                    })

    # 2) 본문 텍스트 완전 일치 검색 (예: 성경 구절을 그대로 복사 붙여넣기 한 경우)
    norm_query = re.sub(r"[^가-힣a-zA-Z0-9]", "", query)
    if len(norm_query) >= 12:
        for key, val in passage_text_store.items():
            norm_text = re.sub(r"[^가-힣a-zA-Z0-9]", "", val.get("text", ""))
            if norm_query in norm_text:
                # 중복 추가 방지
                if any(c["id"] == key for c in candidates):
                    continue
                
                meta = val.get("metadata", {})
                candidates.append({
                    "id": key,
                    "reference": val.get("reference", ""),
                    "book_name": meta.get("book_name", ""),
                    "title": meta.get("title", ""),
                    "summary": meta.get("summary", ""),
                    "tags": meta.get("tags", []),
                    "modern_queries": meta.get("modern_queries", []),
                    "distance": 0.0,
                    "text": val.get("text", "본문 없음"),
                    "is_direct_match": True
                })

    return candidates


# =========================
# 메인
# =========================

def main():
    print("Jina + Qdrant 검색 클라이언트 준비 중...")

    collection = get_qdrant_client()

    with open(PASSAGE_TEXT_STORE_PATH, "r", encoding="utf-8") as f:
        passage_text_store = json.load(f)

    print("검색 준비 완료. Gemma는 검색 시점에 사용하지 않습니다.")

    while True:
        query = input("\n고민을 입력하세요. 종료하려면 q: ").strip()

        if query.lower() in ["q", "quit", "exit"]:
            break

        if not query:
            continue

        candidates = build_hybrid_candidates(
            collection=collection,
            passage_text_store=passage_text_store,
            search_query=query,
        )

        if not candidates:
            print("검색 결과가 없습니다.")
            continue

        final_results = candidates[:FINAL_RESULT_COUNT]

        for rank, candidate in enumerate(final_results, start=1):
            print_result(rank, candidate)


if __name__ == "__main__":
    main()