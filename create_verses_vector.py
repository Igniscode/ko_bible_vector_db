import re
import time

from config import (
    BATCH_SIZE,
    BIBLE_TXT_PATH,
    JINA_EMBED_DIM,
    QDRANT_VERSES_COLLECTION_NAME,
)
from utils import BOOK_MAP, jina_embed_texts, qdrant_ensure_collection, qdrant_upsert_points


def parse_bible_txt(path):
    pattern = re.compile(r"^([가-힣]+)(\d+):(\d+)\s+(.+)$")
    verses = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            m = pattern.match(line)

            if not m:
                print(f"파싱 실패 line {line_no}: {line}")
                continue

            book_abbr, chapter, verse, text = m.groups()
            book_name = BOOK_MAP.get(book_abbr, book_abbr)

            verses.append({
                "book_abbr": book_abbr,
                "book_name": book_name,
                "chapter": int(chapter),
                "verse": int(verse),
                "reference": f"{book_name} {chapter}:{verse}",
                "raw_reference": f"{book_abbr}{chapter}:{verse}",
                "text": text,
                "raw": line,
            })

    return verses


def make_document_text(v):
    return f"""
{v["reference"]}
{v["raw_reference"]}
{v["text"]}
""".strip()


def main():
    qdrant_ensure_collection(QDRANT_VERSES_COLLECTION_NAME, JINA_EMBED_DIM)

    verses = parse_bible_txt(BIBLE_TXT_PATH)
    total = len(verses)

    print(f"업로드 대상 verse: {total}개")

    batch = []

    for idx, v in enumerate(verses, start=1):
        payload = {
            "type": "verse",
            "book_abbr": v["book_abbr"],
            "book_name": v["book_name"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "reference": v["reference"],
            "raw_reference": v["raw_reference"],
            "text": v["text"],
            "raw": v["raw"],
        }

        vector = jina_embed_texts([make_document_text(v)], task="retrieval.passage")[0]

        point = {
            "id": idx,
            "vector": vector,
            "payload": payload,
        }

        batch.append(point)

        if len(batch) >= BATCH_SIZE:
            qdrant_upsert_points(QDRANT_VERSES_COLLECTION_NAME, batch)
            print(f"[{idx}/{total}] verse 업로드 완료")
            batch.clear()
            time.sleep(0.2)

    if batch:
        qdrant_upsert_points(QDRANT_VERSES_COLLECTION_NAME, batch)

    print("verse 업로드 완료")


if __name__ == "__main__":
    main()