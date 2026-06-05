# ko_bible_vector_db

검색어와 관련된 성경 구절을 찾을 수 있는 한국어 성경 벡터 검색 프로젝트입니다.

## 주요 기능

- 자연어 검색어를 기반으로 관련 성경 구절 추천
- 절(verse) / 단락(passage) 단위 결과 표시
- 구절 리더(가독성 높은 본문 보기, 글자 크기 조절, 복사)
- 다크/라이트 테마 전환
- 모바일 드로어 기반 상세 보기
- 프론트 화면의 "제작 정보" 창에서 사용 기술 확인 가능

## 제작에 사용한 것

- 임베딩: Jina Embeddings v3
- 벡터 DB: Qdrant
- 배포: Vercel
- 데이터: 개역한글성경

무료로 제공된 서비스/도구는 Jina, Qdrant, Vercel을 사용했습니다.

## 프로젝트 구조

- `app.py`: Flask 앱 엔트리 포인트
- `search_bible.py`: 검색 로직
- `utils.py`: 임베딩/유틸리티 함수
- `create_bible_vector.py`, `create_verses_vector.py`: 벡터 DB 생성 스크립트
- `static/index.html`: 웹 UI
- `data/`: 성경 원문 및 단락 데이터

## 실행

환경에 맞는 의존성을 설치한 뒤 Flask 앱을 실행하세요.

```bash
pip install -r requirements.txt
python app.py
```

