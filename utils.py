# -*- coding: utf-8 -*-
import importlib
import random
import re
import time

import requests

from config import (
	EMBEDDING_PROVIDER,
	JINA_API_KEY,
	JINA_EMBED_API_URL,
	JINA_EMBED_DIM,
	JINA_EMBED_MODEL,
	JINA_MAX_RETRIES,
	JINA_RETRY_BASE_DELAY,
	JINA_RETRY_MAX_DELAY,
	QDRANT_BASE_URL,
	QDRANT_HEADERS,
)


BOOK_MAP = {
	"창": "창세기", "출": "출애굽기", "레": "레위기", "민": "민수기", "신": "신명기",
	"수": "여호수아", "삿": "사사기", "룻": "룻기",
	"삼상": "사무엘상", "삼하": "사무엘하",
	"왕상": "열왕기상", "왕하": "열왕기하",
	"대상": "역대상", "대하": "역대하",
	"스": "에스라", "느": "느헤미야", "에": "에스더",
	"욥": "욥기", "시": "시편", "잠": "잠언", "전": "전도서", "아": "아가",
	"사": "이사야", "렘": "예레미야", "애": "예레미야애가", "겔": "에스겔",
	"단": "다니엘", "호": "호세아", "욜": "요엘", "암": "아모스", "옵": "오바댜",
	"욘": "요나", "미": "미가", "나": "나훔", "합": "하박국", "습": "스바냐",
	"학": "학개", "슥": "스가랴", "말": "말라기",
	"마": "마태복음", "막": "마가복음", "눅": "누가복음", "요": "요한복음",
	"행": "사도행전", "롬": "로마서", "고전": "고린도전서", "고후": "고린도후서",
	"갈": "갈라디아서", "엡": "에베소서", "빌": "빌립보서", "골": "골로새서",
	"살전": "데살로니가전서", "살후": "데살로니가후서",
	"딤전": "디모데전서", "딤후": "디모데후서", "딛": "디도서", "몬": "빌레몬서",
	"히": "히브리서", "약": "야고보서",
	"벧전": "베드로전서", "벧후": "베드로후서",
	"요일": "요한일서", "요이": "요한이서", "요삼": "요한삼서",
	"유": "유다서", "계": "요한계시록",
}


BIBLE_NAME_MAP = {
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
	"요한계시록": "계", "계시록": "계", "계": "계",
}


def normalize_text(text: str) -> str:
	text = str(text or "").strip()
	return re.sub(r"\s+", " ", text)


def qdrant_request(method, path, payload=None, timeout=60):
	resp = requests.request(
		method=method,
		url=f"{QDRANT_BASE_URL}{path}",
		headers=QDRANT_HEADERS,
		json=payload,
		timeout=timeout,
	)

	if not resp.ok:
		raise RuntimeError(f"Qdrant error {resp.status_code}: {resp.text}")

	return resp.json()


def qdrant_collection_exists(collection_name):
	data = qdrant_request("GET", "/collections")
	collections = data.get("result", {}).get("collections", [])
	return any(c.get("name") == collection_name for c in collections)


def qdrant_ensure_collection(collection_name, vector_size):
	if qdrant_collection_exists(collection_name):
		return

	qdrant_request(
		"PUT",
		f"/collections/{collection_name}",
		{
			"vectors": {
				"size": vector_size,
				"distance": "Cosine",
			}
		},
	)


def qdrant_upsert_points(collection_name, points, wait=True):
	return qdrant_request(
		"PUT",
		f"/collections/{collection_name}/points?wait={'true' if wait else 'false'}",
		{"points": points},
		timeout=120,
	)


def qdrant_search_points(collection_name, query_vector, limit):
	resp = requests.post(
		f"{QDRANT_BASE_URL}/collections/{collection_name}/points/query",
		headers=QDRANT_HEADERS,
		json={"query": query_vector, "limit": limit, "with_payload": True},
		timeout=60,
	)
	resp.raise_for_status()

	body = resp.json()
	points = body.get("result", {}).get("points", [])
	return points if isinstance(points, list) else []


def _get_local_embed_backend_module():
	try:
		return importlib.import_module("local_embed_backend")
	except Exception as e:
		raise RuntimeError(
			"로컬 임베딩 모듈 로딩 실패. requirements-local.txt 설치 후 다시 시도하세요."
		) from e


def jina_embed_text(text: str, task: str):
	if EMBEDDING_PROVIDER == "local":
		backend = _get_local_embed_backend_module()
		return backend.embed_texts([text], task=task)[0]

	if EMBEDDING_PROVIDER != "jina":
		raise RuntimeError(f"지원하지 않는 EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

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


def jina_embed_texts(texts, task="retrieval.passage"):
	if EMBEDDING_PROVIDER == "local":
		backend = _get_local_embed_backend_module()
		return backend.embed_texts(texts, task=task)

	if EMBEDDING_PROVIDER != "jina":
		raise RuntimeError(f"지원하지 않는 EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

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
			resp = requests.post(JINA_EMBED_API_URL, headers=headers, json=payload, timeout=120)

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
