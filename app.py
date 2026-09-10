from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st

from ai_engine import AIEngine
from document_engine import (
    LocalRetriever,
    chunk_records,
    context_to_text,
    extract_uploaded_file,
    reference_labels,
)


st.set_page_config(
    page_title="AI 학습 도우미",
    page_icon="🎓",
    layout="wide",
)



# -------------------------
# Session state
# -------------------------
DEFAULTS = {
    "chunks": [],
    "retriever": None,
    "question": None,
    "question_context": [],
    "oral_eval": None,
    "practice": None,
    "practice_context": [],
    "practice_eval": None,
    "weak_queue": [],
    "ai_call_count": 0,
    "attempts": [],
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _secret(name: str, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_provider_config():
    provider = st.session_state.get("provider", "고려대 API Gateway")

    if provider == "고려대 API Gateway":
        return {
            "provider": provider,
            "api_key": st.session_state.get("user_api_key", "").strip(),
            "base_url": "https://factchat.mindlogic-kr-api.com/v1/gateway",
            "model": st.session_state.get("user_model", "gpt-5.6-luna").strip(),
        }

    return {
        "provider": "OpenAI API",
        "api_key": st.session_state.get("user_api_key", "").strip(),
        "base_url": None,
        "model": st.session_state.get("user_model", "gpt-5.6-luna").strip(),
    }


def get_engine():
    cfg = get_provider_config()

    if not cfg["api_key"]:
        st.error("왼쪽 사이드바에 본인의 API Key를 입력해 주세요.")
        return None

    max_calls = 30
    used = int(st.session_state.get("ai_call_count", 0))
    if used >= max_calls:
        st.error(
            "현재 브라우저 세션에서 AI를 30회 호출했습니다. "
            "새 학습 세션을 시작하려면 페이지를 다시 열어 주세요."
        )
        return None

    return AIEngine(
        cfg["api_key"],
        model=cfg["model"],
        base_url=cfg["base_url"],
    )


def pick_context(query: str, k: int = 7):
    retriever = st.session_state.retriever
    if retriever is None:
        return []
    if query.strip():
        return retriever.search(query, k=k)
    return retriever.representative(k=k)


def safe_api_call(fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        st.session_state.ai_call_count = int(st.session_state.get("ai_call_count", 0)) + 1
        return result
    except Exception as e:
        message = str(e)
        st.error(f"AI 호출 중 오류가 발생했습니다: {message}")
        if "credit_balance_exhausted" in message or "insufficient_quota" in message:
            st.warning("OpenAI API 크레딧이 없습니다. 앱 배포와 화면 사용은 가능하지만 AI 생성 기능은 크레딧 충전 후 작동합니다.")
        else:
            st.info("배포 Secret의 API Key, 모델 이름, 네트워크 상태를 확인해 주세요.")
        return None


def verdict_badge(verdict: str):
    mapping = {
        "correct": ("✅", "정답/충분"),
        "partial": ("🟡", "부분 정답"),
        "incorrect": ("❌", "오답"),
    }
    return mapping.get(verdict, ("ℹ️", verdict))


def parse_number(text: str):
    if not text:
        return None
    s = text.replace(",", "")
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
    if not matches:
        return None
    # 최종 답 입력칸이므로 마지막 숫자를 채점값으로 사용
    try:
        return float(matches[-1])
    except ValueError:
        return None


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("🎓 AI 학습 도우미")
    st.caption("강의자료 기반 구술시험 → 약점 탐지 → 맞춤 문제 → 피드백")

    st.session_state.provider = st.selectbox(
        "AI 제공자",
        ["고려대 API Gateway", "OpenAI API"],
        index=0 if st.session_state.get("provider", "고려대 API Gateway") == "고려대 API Gateway" else 1,
    )

    st.session_state.user_api_key = st.text_input(
        "내 API Key",
        value=st.session_state.get("user_api_key", ""),
        type="password",
        placeholder="본인의 API Key를 입력",
        help=(
            "입력한 키는 이 앱의 현재 브라우저 세션에서 AI 호출에만 사용하며, "
            "GitHub나 학습 DB에 저장하지 않습니다."
        ),
    )

    default_model = st.session_state.get("user_model", "gpt-5.6-luna")
    st.session_state.user_model = st.text_input(
        "모델",
        value=default_model,
        help=(
            "고려대 API 사용 시 학교 API Gateway의 모델 ID를 입력하세요. "
            "예: gpt-5.6-luna"
        ),
    )

    if st.session_state.provider == "고려대 API Gateway":
        st.caption("학교 API Gateway 사용 · 사용량은 입력한 본인 계정의 API 크레딧에서 차감됩니다.")
    else:
        st.caption("OpenAI 직접 API 사용 · 사용량은 입력한 본인 OpenAI API 계정에서 차감됩니다.")

    if st.session_state.user_api_key:
        st.success("개인 API Key 입력됨")
    else:
        st.warning("AI 기능을 사용하려면 본인의 API Key를 입력해 주세요.")

    st.caption(
        f"현재 세션 AI 호출: {int(st.session_state.get('ai_call_count', 0))} / 30"
    )

    st.divider()
    if st.session_state.chunks:
        lecture_count = sum(c["kind"] == "lecture" for c in st.session_state.chunks)
        exam_count = sum(c["kind"] == "exam" for c in st.session_state.chunks)
        st.success(f"자료 준비 완료 · 강의 {lecture_count} chunks / 기출 {exam_count} chunks")
    else:
        st.warning("아직 분석된 자료가 없습니다.")


st.title("AI 학습 도우미")
st.write(
    "강의자료와 기출문제를 기반으로 **구술 질문 → 답변 평가 → 취약 개념 탐지 → 맞춤 문제 → 채점·피드백**을 반복하는 AI 학습 프로토타입입니다."
)
st.caption("공개 링크로 접속한 뒤 각 사용자가 자신의 API Key를 입력해 사용합니다. 다른 사용자의 크레딧이나 학습 기록과 섞이지 않습니다.")

tab_upload, tab_oral, tab_practice, tab_dashboard = st.tabs(
    ["1. 자료 등록", "2. 구술시험", "3. 맞춤 연습", "4. 학습 현황"]
)


# -------------------------
# 1. Upload
# -------------------------
with tab_upload:
    st.subheader("강의자료 / 기출문제 등록")
    st.caption("현재 MVP는 PDF, PPTX, TXT, MD를 지원합니다. 스캔 이미지 PDF는 OCR이 없어 텍스트 추출이 안 될 수 있습니다.")

    col1, col2 = st.columns(2)
    with col1:
        lecture_files = st.file_uploader(
            "강의자료",
            type=["pdf", "pptx", "txt", "md"],
            accept_multiple_files=True,
            key="lecture_upload",
        )
    with col2:
        exam_files = st.file_uploader(
            "기출문제",
            type=["pdf", "pptx", "txt", "md"],
            accept_multiple_files=True,
            key="exam_upload",
        )

    if st.button("자료 분석하기", type="primary", use_container_width=True):
        if not lecture_files and not exam_files:
            st.warning("강의자료 또는 기출문제를 하나 이상 업로드해 주세요.")
        else:
            page_records = []
            failed = []
            with st.spinner("자료에서 텍스트를 추출하고 검색 인덱스를 만드는 중..."):
                for f in lecture_files or []:
                    try:
                        page_records.extend(extract_uploaded_file(f, "lecture"))
                    except Exception as e:
                        failed.append(f"{f.name}: {e}")
                for f in exam_files or []:
                    try:
                        page_records.extend(extract_uploaded_file(f, "exam"))
                    except Exception as e:
                        failed.append(f"{f.name}: {e}")

                chunks = chunk_records(page_records)
                if chunks:
                    st.session_state.chunks = chunks
                    st.session_state.retriever = LocalRetriever(chunks)
                    st.session_state.question = None
                    st.session_state.oral_eval = None
                    st.session_state.practice = None
                    st.session_state.practice_eval = None
                    st.session_state.weak_queue = []
                    st.success(f"완료: {len(page_records)}개 페이지/슬라이드 → {len(chunks)}개 학습 chunk")
                else:
                    st.error("추출된 텍스트가 없습니다. 스캔 PDF라면 텍스트 PDF로 변환해 주세요.")

            for msg in failed:
                st.error(msg)

    if st.session_state.chunks:
        with st.expander("인식된 자료 미리보기"):
            for c in st.session_state.chunks[:5]:
                st.markdown(f"**{c['source']} · {c['page']}p/slide · {c['kind']}**")
                st.text(c["text"][:700] + ("…" if len(c["text"]) > 700 else ""))


# -------------------------
# 2. Oral exam
# -------------------------
with tab_oral:
    st.subheader("AI 구술시험")

    if not st.session_state.chunks:
        st.info("먼저 1번 탭에서 자료를 등록해 주세요.")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            topic = st.text_input(
                "학습 범위 또는 주제",
                placeholder="예: Deal-Grove model, CVD, NMR T1/T2 (비우면 자료 전체)",
            )
        with c2:
            mode = st.selectbox("시험 모드", ["혼합", "개념 중심", "계산 중심"])
        with c3:
            difficulty = st.selectbox("난이도", ["easy", "medium", "hard"], index=1)

        weak_hint = st.session_state.weak_queue[0] if st.session_state.weak_queue else ""

        if weak_hint:
            st.info(f"최근 취약 개념을 다음 질문에 반영할 수 있습니다: **{weak_hint}**")

        if st.button("새 구술 질문 만들기", type="primary"):
            engine = get_engine()
            if engine:
                query = " ".join(x for x in [topic, weak_hint, mode] if x)
                ctx_items = pick_context(query, k=8)
                ctx = context_to_text(ctx_items)
                with st.spinner("강의자료와 기출 스타일을 바탕으로 질문을 만드는 중..."):
                    q = safe_api_call(
                        engine.generate_oral_question,
                        ctx,
                        topic,
                        mode,
                        difficulty,
                        weak_hint,
                    )
                if q:
                    st.session_state.question = q
                    st.session_state.question_context = ctx_items
                    st.session_state.oral_eval = None

        q = st.session_state.question
        if q:
            st.markdown("### 교수 질문")
            st.info(q["question"])
            st.caption(
                f"유형: {q['question_type']} · 난이도: {q['difficulty']} · "
                f"확인 개념: {', '.join(q['target_concepts'])}"
            )
            with st.expander("이 질문이 참고한 자료"):
                st.write(" / ".join(reference_labels(st.session_state.question_context)))

            oral_answer = st.text_area(
                "내 답변",
                height=180,
                placeholder="교수님 앞에서 말하듯 핵심 개념과 이유를 설명해 보세요.",
                key="oral_answer",
            )

            if st.button("답변 제출 및 평가", key="grade_oral"):
                if not oral_answer.strip():
                    st.warning("답변을 입력해 주세요.")
                else:
                    engine = get_engine()
                    if engine:
                        # 채점 시 질문 자체로 관련 문맥을 다시 검색
                        eval_items = pick_context(q["question"] + " " + oral_answer, k=8)
                        eval_ctx = context_to_text(eval_items)
                        with st.spinner("답변의 강점과 취약 개념을 분석하는 중..."):
                            ev = safe_api_call(
                                engine.evaluate_oral_answer,
                                q["question"],
                                oral_answer,
                                eval_ctx,
                            )
                        if ev:
                            st.session_state.oral_eval = ev
                            for wc in ev["weak_concepts"]:
                                if wc not in st.session_state.weak_queue:
                                    st.session_state.weak_queue.append(wc)
                            st.session_state.attempts.append({
                                "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "stage": "oral",
                                "concept": ", ".join(q["target_concepts"]),
                                "score": ev["score"],
                                "verdict": ev["verdict"],
                                "error_type": ev["error_type"],
                                "question": q["question"],
                                "student_answer": oral_answer,
                                "feedback": ev["feedback"],
                                "weak_concepts": ev["weak_concepts"],
                            })

        ev = st.session_state.oral_eval
        if ev:
            icon, label = verdict_badge(ev["verdict"])
            st.markdown(f"### {icon} 평가: {label} · **{ev['score']}점**")

            left, right = st.columns(2)
            with left:
                st.markdown("**잘한 부분**")
                if ev["strengths"]:
                    for x in ev["strengths"]:
                        st.write(f"✓ {x}")
                else:
                    st.write("—")
            with right:
                st.markdown("**보완할 부분**")
                if ev["missing_points"]:
                    for x in ev["missing_points"]:
                        st.write(f"• {x}")
                else:
                    st.write("—")

            st.markdown("**피드백**")
            st.write(ev["feedback"])

            if ev["weak_concepts"]:
                st.warning("취약 개념: " + ", ".join(ev["weak_concepts"]))

            with st.expander("모범 답안 확인"):
                st.write(ev["ideal_answer"])

            st.caption("취약 개념은 자동으로 3번 '맞춤 연습' 탭에 연결됩니다.")


# -------------------------
# 3. Practice
# -------------------------
with tab_practice:
    st.subheader("취약 개념 맞춤 연습")

    if not st.session_state.chunks:
        st.info("먼저 자료를 등록해 주세요.")
    else:
        known_weak = list(dict.fromkeys(st.session_state.weak_queue))
        default_concept = known_weak[0] if known_weak else ""

        p1, p2, p3 = st.columns([2, 1, 1])
        with p1:
            weak_concept = st.text_input(
                "연습할 개념",
                value=default_concept,
                placeholder="구술시험에서 찾은 취약 개념 또는 직접 입력",
            )
        with p2:
            desired_type = st.selectbox("문제 유형", ["자동", "개념 문제", "계산 문제"])
        with p3:
            p_difficulty = st.selectbox(
                "연습 난이도", ["easy", "medium", "hard"], index=1, key="p_diff"
            )

        if known_weak:
            st.caption("현재 취약 개념: " + " · ".join(known_weak[:8]))

        if st.button("맞춤 문제 생성", type="primary", key="make_practice"):
            if not weak_concept.strip():
                st.warning("연습할 개념을 입력해 주세요.")
            else:
                engine = get_engine()
                if engine:
                    ctx_items = pick_context(weak_concept, k=9)
                    ctx = context_to_text(ctx_items)
                    with st.spinner("취약 개념과 기출 스타일을 반영해 문제를 생성하는 중..."):
                        problem = safe_api_call(
                            engine.generate_practice_problem,
                            ctx,
                            weak_concept,
                            desired_type,
                            p_difficulty,
                        )
                    if problem:
                        st.session_state.practice = problem
                        st.session_state.practice_context = ctx_items
                        st.session_state.practice_eval = None

        problem = st.session_state.practice
        if problem:
            st.markdown("### 연습 문제")
            st.info(problem["problem"])
            st.caption(
                f"개념: {problem['concept']} · 유형: {problem['problem_type']} · "
                f"난이도: {problem['difficulty']} · 답 형식: {problem['answer_format']}"
            )
            with st.expander("문제 생성에 참고한 자료"):
                st.write(" / ".join(reference_labels(st.session_state.practice_context)))

            student_final = st.text_input(
                "최종 답",
                placeholder="계산 문제라면 최종 수치와 단위를, 개념 문제라면 핵심 답을 입력",
                key="practice_final",
            )
            student_work = st.text_area(
                "풀이 과정 / 설명 (권장)",
                height=150,
                placeholder="틀렸을 때 정확한 피드백을 받으려면 풀이 과정을 적어 주세요.",
                key="practice_work",
            )

            if st.button("채점하기", key="grade_practice"):
                if not student_final.strip():
                    st.warning("최종 답을 입력해 주세요.")
                else:
                    engine = get_engine()
                    if engine:
                        ctx = context_to_text(st.session_state.practice_context)

                        if problem["grading_mode"] == "numeric" and problem["numeric_answer"] is not None:
                            parsed = parse_number(student_final)
                            if parsed is None:
                                st.warning("최종 답에서 숫자를 인식하지 못했습니다. 예: 16.4 kPa")
                            else:
                                correct = float(problem["numeric_answer"])
                                tol = problem["numeric_tolerance"]
                                if tol is None:
                                    tol = max(abs(correct) * 0.01, 1e-9)
                                tol = float(tol)
                                is_correct = abs(parsed - correct) <= tol

                                if is_correct:
                                    pe = {
                                        "is_correct": True,
                                        "score": 100,
                                        "error_type": "none",
                                        "reason": f"수치가 허용 오차 ±{tol:g} 범위 안에 있습니다.",
                                        "feedback": "정답입니다. 풀이 과정에서 사용한 식과 단위까지 다시 확인하면 더 안정적으로 기억할 수 있습니다.",
                                        "weak_concepts": [],
                                    }
                                else:
                                    with st.spinner("정답 판정 완료. 틀린 이유를 풀이 과정과 비교하는 중..."):
                                        nf = safe_api_call(
                                            engine.explain_numeric_error,
                                            problem,
                                            student_final,
                                            student_work,
                                            ctx,
                                        )
                                    if nf is None:
                                        pe = None
                                    else:
                                        pe = {
                                            "is_correct": False,
                                            "score": 0,
                                            "error_type": nf["error_type"],
                                            "reason": nf["reason"],
                                            "feedback": nf["feedback"],
                                            "weak_concepts": nf["weak_concepts"],
                                        }
                        else:
                            with st.spinner("답안을 강의자료 기준으로 채점하는 중..."):
                                pe = safe_api_call(
                                    engine.evaluate_practice_answer,
                                    problem,
                                    student_final,
                                    student_work,
                                    ctx,
                                )

                        if pe:
                            st.session_state.practice_eval = pe
                            for wc in pe["weak_concepts"]:
                                if wc not in st.session_state.weak_queue:
                                    st.session_state.weak_queue.append(wc)

                            st.session_state.attempts.append({
                                "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "stage": "practice",
                                "concept": problem["concept"],
                                "score": pe["score"],
                                "verdict": "correct" if pe["is_correct"] else "incorrect",
                                "error_type": pe["error_type"],
                                "question": problem["problem"],
                                "student_answer": student_final,
                                "feedback": pe["feedback"],
                                "weak_concepts": pe["weak_concepts"],
                            })

        pe = st.session_state.practice_eval
        if pe:
            if pe["is_correct"]:
                st.success(f"✅ 정답입니다 · {pe['score']}점")
            else:
                st.error(f"❌ 오답입니다 · {pe['score']}점")

            st.markdown("**판정 이유**")
            st.write(pe["reason"])
            st.markdown("**피드백**")
            st.write(pe["feedback"])

            if pe["weak_concepts"]:
                st.warning("추가로 확인할 개념: " + ", ".join(pe["weak_concepts"]))

            with st.expander("정답 및 기준 풀이"):
                st.markdown("**기준 답안**")
                st.write(problem["reference_answer"])
                st.markdown("**풀이**")
                st.write(problem["solution"])


# -------------------------
# 4. Dashboard
# -------------------------
with tab_dashboard:
    st.subheader("학습 현황")
    st.caption("이 기록은 현재 사용자의 브라우저 세션에만 유지되며 다른 사용자와 공유되지 않습니다.")

    attempts = list(st.session_state.get("attempts", []))

    if not attempts:
        st.info("아직 채점 기록이 없습니다. 구술시험이나 맞춤 문제를 풀면 여기에 누적됩니다.")
    else:
        df = pd.DataFrame(attempts)
        # 최신 기록이 위로 오도록 표시
        display_df = df.iloc[::-1].reset_index(drop=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("총 응답", len(df))
        c2.metric("평균 점수", f"{df['score'].mean():.1f}")
        c3.metric("최근 점수", int(df.iloc[-1]["score"]))

        st.markdown("### 오류 유형")
        error_counts = (
            df[df["error_type"] != "none"]["error_type"]
            .value_counts()
            .rename_axis("오류 유형")
            .reset_index(name="횟수")
        )
        if not error_counts.empty:
            st.bar_chart(error_counts.set_index("오류 유형"))
        else:
            st.success("기록된 오류가 없습니다.")

        from collections import Counter
        weak_counter = Counter()
        for items in df["weak_concepts"]:
            if isinstance(items, list):
                weak_counter.update([x for x in items if x])

        if weak_counter:
            st.markdown("### 반복적으로 나타난 취약 개념")
            weak_df = pd.DataFrame(
                weak_counter.most_common(10), columns=["개념", "언급 횟수"]
            ).set_index("개념")
            st.bar_chart(weak_df)

        st.markdown("### 최근 학습 기록")
        show_cols = [
            "created_at",
            "stage",
            "concept",
            "score",
            "verdict",
            "error_type",
            "question",
        ]
        st.dataframe(
            display_df[show_cols].head(30),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("현재 학습 기록 초기화"):
            st.session_state.attempts = []
            st.session_state.weak_queue = []
            st.session_state.ai_call_count = 0
            st.success("현재 브라우저 세션의 학습 기록을 초기화했습니다.")
            st.rerun()


st.divider()
st.caption(
    "MVP 설계: 자료는 로컬에서 텍스트 추출/검색하고, 실제 AI 호출에는 관련 chunk만 전달합니다. "
    "사용자의 API Key는 현재 세션에서만 사용하며, 앱은 학습 기록을 서버 DB에 저장하지 않습니다."
)
