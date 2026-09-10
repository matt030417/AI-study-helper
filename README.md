# AI 학습 도우미 — Deploy Ready

강의자료와 기출문제를 기반으로 구술시험을 진행하고, 답변에서 취약 개념을 탐지하여
맞춤 개념/계산 문제와 피드백을 제공하는 Streamlit + OpenAI API 프로토타입입니다.

## 핵심 흐름

자료 업로드 → 관련 자료 검색 → 구술 질문 → 답변 평가 → 취약 개념 →
맞춤 문제 → 정오 판정 → 오답 피드백 → 학습 현황

## 배포

자세한 내용은 `DEPLOY_AFTER_HOMEPAGE_KR.md`를 확인하세요.

## 보안

공개 배포에서는 OpenAI API Key를 코드나 GitHub에 넣지 않고
Streamlit Community Cloud의 Secrets 기능에 저장합니다.

---

# OralLoop AI — 강의자료 기반 AI 구술시험 프로토타입

강의자료와 기출문제를 넣으면 다음 학습 루프를 실제로 수행하는 Streamlit MVP입니다.

**자료 업로드 → 구술 질문 → 답변 평가 → 취약 개념 탐지 → 맞춤 문제 생성 → 채점/오답 피드백 → 학습 기록**

## 1. 현재 구현된 기능

- PDF / PPTX / TXT / MD 강의자료 업로드
- 기출문제를 강의자료와 별도로 구분해서 업로드
- 로컬 TF-IDF 검색을 이용한 간단한 RAG
- 강의자료/기출 스타일 기반 구술 질문 생성
- 학생 답변 채점: 정답 / 부분 정답 / 오답
- 강점, 빠진 핵심, 취약 개념, 오류 유형 분석
- 취약 개념을 자동으로 맞춤 연습에 연결
- 개념 문제 / 계산 문제 생성
- 계산 문제는 가능한 경우 Python 로직으로 수치 정답 판정
- 오답 원인과 피드백 생성
- SQLite에 학습 기록 저장
- 평균 점수, 오류 유형, 반복 취약 개념 대시보드

## 2. 실행 방법

Python 3.11+ 권장.

```bash
cd ai_oral_exam_prototype
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 열리면 왼쪽 사이드바에 OpenAI API Key를 입력합니다.

환경변수로 넣고 싶다면:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.6-luna"
streamlit run app.py
```

## 3. 추천 데모 순서

1. 강의자료 PDF 1개와 기출문제 PDF 1개 업로드
2. `자료 분석하기`
3. `구술시험` 탭에서 주제를 입력하거나 자료 전체로 질문 생성
4. 일부러 핵심을 하나 빠뜨린 답변 제출
5. AI가 `부분 정답 + 취약 개념`을 잡아내는지 확인
6. `맞춤 연습` 탭으로 이동
7. 자동으로 연결된 취약 개념에 대해 계산/개념 문제 생성
8. 답 입력 후 채점 및 피드백 확인
9. `학습 현황`에서 오류 유형과 취약 개념 누적 확인

## 4. 프로젝트 구조

```text
ai_oral_exam_prototype/
├── app.py               # Streamlit UI + 학습 흐름
├── ai_engine.py         # 질문/채점/문제생성/피드백 AI
├── document_engine.py   # PDF/PPTX 추출 + chunking + local RAG
├── storage.py           # SQLite 학습기록
├── requirements.txt
├── .env.example
└── README.md
```

## 5. 설계 포인트

### RAG
전체 강의자료를 매번 AI에게 보내지 않습니다. 문서는 로컬에서 chunk로 나누고 TF-IDF로 관련 부분을 찾은 뒤, 관련 chunk만 AI에 전달합니다.

### AI 역할 분리
하나의 거대한 프롬프트가 아니라 다음 역할을 분리했습니다.

- Oral Question Generator
- Oral Answer Evaluator
- Weakness Analyzer(평가 결과 내부)
- Practice Problem Generator
- Practice Answer Evaluator
- Numeric Error Coach

같은 OpenAI API를 사용하지만 역할별 프롬프트와 출력 스키마를 나누어 결과를 안정적으로 연결합니다.

### 계산 문제
문제 생성 AI가 `grading_mode=numeric`으로 만든 경우:
- 최종 숫자 정답 판정: 프로그램
- 왜 틀렸는지 설명: AI

이렇게 분리해 단순 LLM 채점보다 안정적으로 만들었습니다.

## 6. 현재 MVP의 한계

- 스캔 이미지 PDF는 OCR을 하지 않습니다.
- 수식이 이미지로만 들어간 PDF/PPTX는 텍스트로 인식되지 않을 수 있습니다.
- 현재 구술은 텍스트 입력입니다. 음성 입력은 다음 버전에 추가할 수 있습니다.
- TF-IDF RAG는 가볍고 무료이지만, 고도화 시 embedding/vector DB로 교체할 수 있습니다.
- AI 채점은 교육용 보조 기능이며 절대적인 채점 기준으로 사용하면 안 됩니다.

## 7. 다음 버전 후보

- 마이크 음성 입력 + STT
- 실제 구술처럼 제한시간/꼬리질문 모드
- 강의자료 원본 페이지 미리보기
- 과목별/챕터별 concept graph
- 난이도 자동 조절
- spaced repetition
- 교수별 기출 스타일 분석
- embedding + vector database
- 로그인/사용자별 DB
- 배포(Streamlit Community Cloud 등)

## 8. API 모델

기본값은 비용을 고려해 `gpt-5.6-luna`로 두었습니다. 사이드바에서 다른 접근 가능한 모델 ID로 바꿀 수 있습니다.

OpenAI Responses API의 Structured Outputs를 사용하도록 작성되어 있습니다.
