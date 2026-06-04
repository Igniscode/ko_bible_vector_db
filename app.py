# -*- coding: utf-8 -*-
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import search_bible


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SearchRequest(BaseModel):
	query: str = Field(min_length=1)
	top_k: int = Field(default=5, ge=1, le=10)


class SearchResponse(BaseModel):
	intent: str
	expanded_query: str
	results: list


app = FastAPI(title="Bible VDB UI Server")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


_collection = None
_passage_text_store = None


def _ensure_loaded():
	global _collection
	global _passage_text_store

	if _collection is None:
		_collection = search_bible.get_qdrant_client()

	if _passage_text_store is None:
		with open(search_bible.PASSAGE_TEXT_STORE_PATH, "r", encoding="utf-8") as f:
			_passage_text_store = json.load(f)


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
def search_api(req: SearchRequest):
	_ensure_loaded()

	query = req.query.strip()
	if not query:
		raise HTTPException(status_code=400, detail="query is required")

	intent = search_bible.detect_intent(query)
	expanded = search_bible.expand_query(query, intent)

	# 1) 구절 직접 검색 여부 확인
	direct_candidates = search_bible.get_direct_candidates(query, _passage_text_store)

	# 2) 하이브리드 검색 후보군 획득
	candidates = search_bible.build_hybrid_candidates(
		collection=_collection,
		passage_text_store=_passage_text_store,
		search_query=expanded,
	)

	# 3) 구절 직접 매칭 결과가 있으면 후보군 처음에 병합 (id 기준 중복 제거)
	if direct_candidates:
		merged = {c["id"]: c for c in candidates}
		for dc in direct_candidates:
			merged[dc["id"]] = dc
		candidates = list(merged.values())

	final_results = search_bible.rerank_by_rules(
		candidates,
		intent=intent,
		final_count=req.top_k,
	)

	return {
		"intent": intent,
		"expanded_query": expanded,
		"results": final_results,
	}

