# -*- coding: utf-8 -*-
import json
import os
import time
import threading
import copy
from collections import defaultdict, deque, OrderedDict
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import search_bible


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

STATIC_PAGE_API_KEY = "fc5b4866e43b0390d8f90280191dc480"
SEARCH_QUERY_MAX_LEN = 200
SEARCH_RATE_LIMIT_PER_MIN = 10
SEARCH_CACHE_TTL_SEC = 180
SEARCH_CACHE_MAX_ITEMS = 500


class SearchRequest(BaseModel):
	query: str = Field(min_length=1, max_length=SEARCH_QUERY_MAX_LEN)
	top_k: int = Field(default=5, ge=1, le=10)


class SearchResponse(BaseModel):
	results: list


app = FastAPI(title="Bible VDB UI Server")


def _origin_matches_host(value: str, expected_scheme: str, expected_host: str) -> bool:
	if not value:
		return False

	try:
		parsed = urlparse(value)
	except Exception:
		return False

	if not parsed.scheme or not parsed.netloc:
		return False

	return parsed.scheme == expected_scheme and parsed.netloc == expected_host


@app.middleware("http")
async def api_same_origin_guard(request: Request, call_next):
	# API는 브라우저의 same-origin 요청만 허용한다.
	if request.url.path.startswith("/api/"):
		host = request.headers.get("host", "")
		scheme = request.url.scheme

		origin = request.headers.get("origin", "")
		referer = request.headers.get("referer", "")
		sec_fetch_site = request.headers.get("sec-fetch-site", "")
		api_key = request.headers.get("x-static-api-key", "")

		origin_ok = _origin_matches_host(origin, scheme, host)
		referer_ok = _origin_matches_host(referer, scheme, host)
		fetch_site_ok = sec_fetch_site in ("", "same-origin", "same-site")
		api_key_ok = bool(STATIC_PAGE_API_KEY) and (api_key == STATIC_PAGE_API_KEY)

		if not ((origin_ok or referer_ok) and fetch_site_ok and api_key_ok):
			return JSONResponse(
				status_code=403,
				content={"detail": "API 호출이 허용되지 않은 요청입니다."},
			)

	response = await call_next(request)
	return response

app.mount("/static", StaticFiles(directory="static"), name="static")


_collection = None
_passage_text_store = None
_rate_buckets = defaultdict(deque)
_rate_lock = threading.Lock()
_search_cache = OrderedDict()
_cache_lock = threading.Lock()


def _ensure_loaded():
	global _collection
	global _passage_text_store

	if _collection is None:
		_collection = search_bible.get_qdrant_client()

	if _passage_text_store is None:
		with open(search_bible.PASSAGE_TEXT_STORE_PATH, "r", encoding="utf-8") as f:
			_passage_text_store = json.load(f)


def _get_client_ip(request: Request) -> str:
	# 프록시 환경에서는 첫 번째 X-Forwarded-For를 우선 사용한다.
	xff = request.headers.get("x-forwarded-for", "")
	if xff:
		first = xff.split(",")[0].strip()
		if first:
			return first

	if request.client and request.client.host:
		return request.client.host

	return "unknown"


def _check_rate_limit(ip: str):
	now = time.time()
	window_start = now - 60.0

	with _rate_lock:
		bucket = _rate_buckets[ip]

		while bucket and bucket[0] < window_start:
			bucket.popleft()

		if len(bucket) >= SEARCH_RATE_LIMIT_PER_MIN:
			retry_after = max(1, int(60 - (now - bucket[0])))
			raise HTTPException(
				status_code=429,
				detail=f"요청이 너무 많습니다. 잠시 후 다시 시도하세요. (retry_after={retry_after}s)",
			)

		bucket.append(now)


def _make_cache_key(query: str, top_k: int) -> str:
	return f"{query}\n{top_k}"


def _cache_get(key: str):
	now = time.time()

	with _cache_lock:
		item = _search_cache.get(key)
		if item is None:
			return None

		expires_at, value = item
		if expires_at <= now:
			del _search_cache[key]
			return None

		_search_cache.move_to_end(key)
		return copy.deepcopy(value)


def _cache_set(key: str, value):
	now = time.time()
	expires_at = now + SEARCH_CACHE_TTL_SEC

	with _cache_lock:
		_search_cache[key] = (expires_at, copy.deepcopy(value))
		_search_cache.move_to_end(key)

		# 오래된 항목 제거
		expired_keys = [k for k, (exp, _) in _search_cache.items() if exp <= now]
		for k in expired_keys:
			_search_cache.pop(k, None)

		# 용량 제한 초과 시 LRU부터 제거
		while len(_search_cache) > SEARCH_CACHE_MAX_ITEMS:
			_search_cache.popitem(last=False)


@app.get("/")
def index():
	return FileResponse(
		"static/index.html",
		headers={
			"Cache-Control": "no-cache, no-store, must-revalidate",
			"Pragma": "no-cache",
			"Expires": "0"
		}
	)


@app.post("/api/search", response_model=SearchResponse)
def search_api(req: SearchRequest, request: Request):
	client_ip = _get_client_ip(request)
	_check_rate_limit(client_ip)

	_ensure_loaded()

	query = req.query.strip()
	if not query:
		raise HTTPException(status_code=400, detail="query is required")

	if len(query) > SEARCH_QUERY_MAX_LEN:
		raise HTTPException(
			status_code=400,
			detail=f"query 길이는 최대 {SEARCH_QUERY_MAX_LEN}자까지 허용됩니다.",
		)

	cache_key = _make_cache_key(query, req.top_k)
	cached = _cache_get(cache_key)
	if cached is not None:
		return cached

	try:
		# 1) 구절 직접 검색 여부 확인
		direct_candidates = search_bible.get_direct_candidates(query, _passage_text_store)

		# 2) 하이브리드 검색 후보군 획득
		candidates = search_bible.build_hybrid_candidates(
			collection=_collection,
			passage_text_store=_passage_text_store,
			search_query=query,
		)

		# 3) 구절 직접 매칭 결과를 앞쪽에 두고 id 기준 중복 제거
		if direct_candidates:
			seen = set()
			merged = []

			for dc in direct_candidates:
				dc_id = dc.get("id")
				if dc_id in seen:
					continue
				seen.add(dc_id)
				merged.append(dc)

			for c in candidates:
				c_id = c.get("id")
				if c_id in seen:
					continue
				seen.add(c_id)
				merged.append(c)

			candidates = merged

		final_results = candidates[:req.top_k]

		response_payload = {
			"results": final_results,
		}

		_cache_set(cache_key, response_payload)
		return response_payload

	except HTTPException:
		raise
	except Exception as e:
		print(f"[search_api] internal error: {e}")
		raise HTTPException(
			status_code=500,
			detail="요청을 처리하는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
		)

