"""Two study workspaces; authentication and project management remain in app.py."""
from __future__ import annotations

import hashlib
from uuid import uuid4

import fitz
import pandas as pd
import streamlit as st

from cloud_storage import delete_project_file, save_project_file
from document_engine import (context_to_text, extract_uploaded_file, pages_to_images,
                             pptx_to_pdf, render_pdf_page_png, solution_file_to_images)
from study_core import (cited_items, file_key, fit_context, parse_pages, reader_id,
                        scoped_records)


class MemoryFile:
    def __init__(self, name, raw):
        self.name, self.raw = name, raw

    def getvalue(self):
        return self.raw


def scope() -> str:
    values = [st.session_state.get(k, "") for k in
              ("cloud_user_id", "current_project_id")]
    return hashlib.sha256(repr(values).encode()).hexdigest()[:12]


def material_upload(kind: str, refresh, optional: bool = False):
    label = "강의자료" if kind == "lecture" else "기출문제"
    if not st.session_state.get("current_project_id"):
        st.info("먼저 학습할 과목을 선택해 주세요.")
        return
    saved = [m for m in st.session_state.current_project_files if m["kind"] == kind]
    optional_label = "선택 · " if optional else ""
    with st.expander(f"{label} 업로드·관리 ({optional_label}{len(saved)}개)", expanded=not saved and not optional):
        st.caption("PDF·PPTX·TXT·MD 지원 · 저장한 자료는 현재 과목에 연결됩니다.")
        with st.form(f"workspace_upload_{kind}_{scope()}", clear_on_submit=True):
            files = st.file_uploader(label, type=["pdf", "pptx", "txt", "md"],
                                     accept_multiple_files=True, key=f"workspace_files_{kind}_{scope()}")
            submitted = st.form_submit_button("저장하고 열기", type="primary")
        if submitted:
            if not files:
                st.warning("파일을 하나 이상 선택해 주세요.")
            else:
                errors, saved_count = [], 0
                with st.spinner(f"{label}를 현재 과목에 저장하는 중..."):
                    for upload in files:
                        try:
                            if len(upload.getvalue()) > 50 * 1024 * 1024:
                                raise ValueError("파일 한 개는 50MB 이하여야 합니다.")
                            if not extract_uploaded_file(upload, kind):
                                raise ValueError("인식할 수 있는 페이지나 텍스트가 없습니다.")
                            save_project_file(st.session_state.cloud_client,
                                              st.session_state.cloud_user_id,
                                              st.session_state.current_project_id,
                                              None,
                                              upload.name, kind, upload.getvalue())
                            saved_count += 1
                        except Exception as exc:
                            errors.append(f"{upload.name}: {exc}")
                    if saved_count:
                        refresh(force=True)
                for error in errors:
                    st.error(error)
                if saved_count and not errors:
                    st.rerun()
                elif saved_count:
                    st.success(f"{saved_count}개 파일을 저장했습니다. 실패한 파일은 다시 올려 주세요.")
        for meta in saved:
            name_col, delete_col = st.columns([5, 1])
            name_col.write(meta.get("display_name", meta["file_name"]))
            if delete_col.button("삭제", key=f"workspace_delete_{meta['id']}_{scope()}"):
                try:
                    delete_project_file(st.session_state.cloud_client,
                                        st.session_state.cloud_user_id, meta["id"])
                    refresh(force=True)
                    st.rerun()
                except Exception as exc:
                    st.error(f"파일을 삭제하지 못했습니다: {exc}")


def _pdf_for(meta: dict) -> bytes | None:
    if meta["suffix"] == "pdf":
        return meta["raw"]
    if meta["suffix"] != "pptx":
        return None
    key = hashlib.sha256(meta["raw"]).hexdigest()
    cache = st.session_state.preview_pdfs
    if key not in cache:
        try:
            cache[key] = pptx_to_pdf(meta["raw"])
        except Exception as exc:
            cache[key] = str(exc)
    value = cache[key]
    if isinstance(value, str):
        st.info(value)
        return None
    return value


def _page_count(meta: dict, records: list[dict]) -> int:
    if meta["suffix"] == "pdf":
        with fitz.open(stream=meta["raw"], filetype="pdf") as pdf:
            return len(pdf)
    return max((int(r["page"]) for r in records), default=1)


def _pick_file(kind: str, widget: str, compact: bool = False):
    keys = [k for k, m in st.session_state.source_files.items() if m["kind"] == kind]
    if not keys:
        return None
    names = {k: st.session_state.source_files[k]["name"] for k in keys}
    selected = st.selectbox("강의자료 선택" if kind == "lecture" else "기출문제 선택",
                            keys, format_func=names.get,
                            label_visibility="collapsed" if compact else "visible",
                            key=f"workspace_{widget}_source_{scope()}")
    return st.session_state.source_files[selected]


def _related_lectures(query: str, k: int = 7) -> list[dict]:
    retriever = st.session_state.retriever
    if not retriever:
        return []
    return (retriever.search(query, k=k, kind="lecture") if query.strip()
            else retriever.representative(k=k, kind="lecture"))


def _images(pdf: bytes | None, pages: list[int], name: str, vision: bool):
    return pages_to_images(pdf, pages, name) if pdf and vision else []


def _needs_vision(records):
    return any("추출 가능한 텍스트가 없습니다" in r["text"] or
               "텍스트가 없는 슬라이드" in r["text"] for r in records)


def _move_reader_page(key: str, offset: int, total: int):
    st.session_state[key] = max(1, min(total, int(st.session_state.get(key, 1)) + offset))


def render_reader(get_engine, call, evidence, refresh):
    st.markdown('''<style>
    .st-key-reader_pager { width:198px !important; max-width:100%; margin-left:auto; }
    .st-key-reader_pager [data-testid="stHorizontalBlock"] { gap:6px; align-items:center; }
    .st-key-reader_pager .stButton button { width:40px !important; height:40px; min-height:40px; padding:0; border-radius:11px; font-size:25px; line-height:1; }
    .st-key-reader_pager [data-testid="stPopover"] button { height:40px; min-height:40px; padding:0 10px; border-radius:11px; font-size:14px; }
    .st-key-reader_document { padding:10px !important; border-radius:16px; overflow:hidden; }
    .st-key-reader_document [data-testid="stImage"] { margin:0; }
    .st-key-reader_document img { display:block; width:100%; height:auto; object-fit:contain; }
    .st-key-reader_explanation, .st-key-reader_chat_panel { border-radius:16px; border:1px solid var(--st-border-color,#e6e8ed); padding:18px !important; }
    .st-key-reader_explanation p, .st-key-reader_chat p { line-height:1.65; }
    .st-key-reader_tools .stButton button, .st-key-reader_tools [data-testid="stPopover"] button { height:36px; min-height:36px; padding:0 10px; border-radius:10px; }
    .st-key-reader_tools [data-testid="stHorizontalBlock"] { align-items:center; }
    .st-key-reader_chat [data-testid="stChatMessage"] { padding:6px 0; background:transparent; }
    </style>''', unsafe_allow_html=True)
    material_upload("lecture", refresh)
    sources = [k for k, m in st.session_state.source_files.items() if m["kind"] == "lecture"]
    if not sources:
        st.info("강의자료를 올리면 왼쪽에 원본, 오른쪽에 페이지별 설명과 채팅이 표시됩니다.")
        return
    selector, pager = st.columns([6, 2], gap="small", vertical_alignment="center")
    with selector:
        meta = _pick_file("lecture", "reader", compact=True)
    records = scoped_records(st.session_state.page_records, "lecture", meta["name"])
    total = _page_count(meta, records)
    raw_id = hashlib.sha256(meta["name"].encode() + meta["raw"]).hexdigest()[:12]
    cursor = f"reader_page_{scope()}_{raw_id}"
    if cursor not in st.session_state:
        st.session_state[cursor] = 1
    st.session_state[cursor] = max(1, min(total, int(st.session_state[cursor])))
    page = int(st.session_state[cursor])
    with pager:
        with st.container(key="reader_pager"):
            prev, middle, nxt = st.columns([1, 2.2, 1], gap="small", vertical_alignment="center")
            prev.button("‹", help="이전 페이지", key=f"reader_prev_{scope()}_{raw_id}",
                        disabled=page == 1, width="stretch",
                        on_click=_move_reader_page, args=(cursor, -1, total))
            with middle:
                with st.popover(f"{page} / {total}", width="stretch"):
                    st.number_input("페이지로 이동", min_value=1, max_value=total, step=1, key=cursor)
            nxt.button("›", help="다음 페이지", key=f"reader_next_{scope()}_{raw_id}",
                       disabled=page == total, width="stretch",
                       on_click=_move_reader_page, args=(cursor, 1, total))
    pages = [page]
    selected = scoped_records(records, "lecture", pages=pages)
    context = fit_context(selected)
    pdf = _pdf_for(meta)
    with st.container(key="reader_workspace"):
        left, right = st.columns([2, 3], gap="large", vertical_alignment="top")
        with left:
            st.markdown("**강의자료**")
            with st.container(border=True, key="reader_document"):
                if pdf:
                    try:
                        # Natural image ratio; no fixed-height wrapper under the slide.
                        st.image(render_pdf_page_png(pdf, page), width="stretch")
                    except Exception as exc:
                        st.warning(f"원본을 표시하지 못했습니다: {exc}")
                else:
                    for record in selected:
                        st.text(record["text"])
            with st.expander("원본 파일"):
                st.download_button("내려받기", meta["raw"], file_name=meta["name"],
                                   key=f"reader_download_{scope()}_{raw_id}")
        with right:
            with st.container(key="reader_tools"):
                title, settings, redo = st.columns([7, 1, 1], gap="small", vertical_alignment="center")
                title.markdown("**해석·설명**")
                with settings:
                    with st.popover("⚙", help="설명 설정"):
                        vision = st.checkbox("그림·수식도 함께 읽기", value=True,
                                             key=f"reader_vision_{scope()}_{raw_id}")
                        st.caption("그림·수식 읽기에는 이미지 입력을 지원하는 모델이 필요합니다.")
                sid = reader_id(st.session_state.cloud_user_id, st.session_state.current_project_id,
                                "", meta["name"], meta["raw"], pages, vision)
                regenerate = redo.button("↻", help="현재 페이지 설명 새로 만들기", key=f"reader_refresh_{sid}")
            blocked = _needs_vision(selected) and (not vision or not pdf)
            attempts = st.session_state.setdefault("reader_auto_attempts", {})
            configuration = repr((st.session_state.get("provider"), st.session_state.get("user_model"),
                                  st.session_state.get("user_api_key", "")))
            attempt_id = sid + hashlib.sha256(configuration.encode()).hexdigest()[:12]
            newly_visible = st.session_state.get("reader_last_visible") != attempt_id
            st.session_state.reader_last_visible = attempt_id
            note = st.session_state.reader_notes.get(sid)
            automatic = (not note and bool(st.session_state.get("user_api_key"))
                         and (newly_visible or attempt_id not in attempts))
            with st.container(height=460, border=True, key="reader_explanation"):
                if (regenerate or automatic) and not blocked:
                    attempts[attempt_id] = True
                    engine = get_engine()
                    if engine:
                        with st.spinner(f"{page}페이지 설명을 불러오는 중..."):
                            try:
                                result = call(engine.explain_material,
                                              context_to_text(context, max_chars=32000),
                                              _images(pdf, pages, meta["name"], vision))
                            except Exception as exc:
                                st.error(f"페이지를 읽지 못했습니다: {exc}")
                                result = None
                        if result:
                            note = {**result, "context": context}
                            st.session_state.reader_notes[sid] = note
                if note:
                    st.caption(f"{page}페이지")
                    st.markdown("**한국어 해석**")
                    st.markdown(note["translation"])
                    st.markdown("**개념·수식·그림 설명**")
                    st.markdown(note["explanation"])
                    evidence(note["context"], note.get("evidence_ids"), title="근거 페이지")
                elif blocked:
                    st.caption("이 페이지는 텍스트를 추출할 수 없습니다. 설명 설정에서 그림 읽기를 켜고 PDF를 사용해 주세요.")
                elif not st.session_state.get("user_api_key"):
                    st.caption("왼쪽에서 API를 연결하면 현재 페이지의 설명을 자동으로 불러옵니다.")
                else:
                    st.caption("설명을 불러오지 못했습니다. API 연결을 확인하거나 ↻로 다시 시도해 주세요.")
            st.markdown("**이 페이지에 질문하기**")
            messages = st.session_state.reader_chats.get(sid, [])
            with st.container(border=True, key="reader_chat_panel"):
                with st.container(height=180, border=False, key="reader_chat"):
                    if not messages:
                        st.caption("수식의 의미나 그림에서 이해되지 않는 부분을 질문해 보세요.")
                    for index, message in enumerate(messages):
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                            if message["role"] == "assistant":
                                evidence(message.get("context", []), message.get("evidence_ids"),
                                         title=f"답변 근거 {index + 1}")
                prompt = st.chat_input("이 페이지에 대해 질문하세요", key=f"reader_chat_input_{sid}", disabled=blocked)
            if prompt:
                engine = get_engine()
                if engine:
                    ctx = fit_context(selected + _related_lectures(prompt))
                    with st.spinner("강의자료에서 답을 찾는 중..."):
                        try:
                            answer = call(engine.chat_material, prompt, context_to_text(ctx, max_chars=32000),
                                          messages, note.get("explanation", "") if note else "",
                                          _images(pdf, pages, meta["name"], vision))
                        except Exception as exc:
                            st.error(f"페이지를 읽지 못했습니다: {exc}")
                            answer = None
                    if answer:
                        messages.extend([{"role": "user", "content": prompt},
                                         {"role": "assistant", "content": answer["answer"],
                                          "evidence_ids": answer["evidence_ids"], "context": ctx}])
                        st.session_state.reader_chats[sid] = messages
                        st.rerun()
                    else:
                        st.caption("질문을 전송하지 못했습니다: " + prompt)


def _save_field(index: int, field: str, key: str):
    value = st.session_state.get(key)
    if field == "file":
        value = {"name": value.name, "raw": value.getvalue()} if value else None
    st.session_state.exam_drafts.setdefault(str(index), {})[field] = value


def _answer_form(index, question, batch_id):
    draft = st.session_state.exam_drafts.setdefault(str(index), {})
    locked = str(index) in st.session_state.exam_results
    for field in ["answer", "work"]:
        key = f"exam_{batch_id}_{index}_{field}"
        if key not in st.session_state:
            st.session_state[key] = draft.get(field, "")
        if field == "answer":
            st.text_input("최종 답", key=key, disabled=locked,
                          on_change=_save_field, args=(index, field, key),
                          placeholder="답과 단위를 입력하거나 아래에 손글씨 풀이를 첨부하세요")
        else:
            st.text_area("풀이 과정 (선택)", key=key, height=110, disabled=locked,
                         on_change=_save_field, args=(index, field, key))
    file_widget = f"exam_{batch_id}_{index}_file"
    st.file_uploader("손글씨 풀이 PDF·사진 (선택)", type=["pdf", "jpg", "jpeg", "png"],
                     key=file_widget, disabled=locked, on_change=_save_field,
                     args=(index, "file", file_widget),
                     help="풀이 파일은 채점 때 AI에 전송됩니다. PDF는 앞 3페이지를 읽습니다.")
    if draft.get("file"):
        st.caption("저장된 풀이 첨부: " + draft["file"]["name"])
        if not locked and st.button("첨부 빼기", key=f"exam_remove_file_{batch_id}_{index}"):
            draft["file"] = None
            st.session_state.pop(file_widget, None)
            st.rerun()


def _numeric_check(question, answer):
    # Unit-bearing answers go to the model so 1 m and 100 cm can be treated as equivalent.
    import math
    import re
    if question.get("grading_mode") != "numeric" or question.get("expected_unit"):
        return None
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", answer.strip()):
        return None
    try:
        correct, parsed = float(question["numeric_answer"]), float(answer.strip())
        tolerance = question.get("numeric_tolerance")
        tolerance = abs(float(tolerance)) if tolerance is not None else max(abs(correct) * .01, 1e-9)
        if not all(math.isfinite(x) for x in [correct, parsed, tolerance]):
            return None
        return {"is_correct": abs(parsed - correct) <= tolerance, "parsed": parsed,
                "correct": correct, "tolerance": tolerance, "unit": ""}
    except (ValueError, TypeError):
        return None


def _grade(indices, get_engine, call, record_attempt):
    batch = st.session_state.exam_set
    pending = [i for i in indices if str(i) not in st.session_state.exam_results]
    missing = [i + 1 for i in pending if not any(st.session_state.exam_drafts.get(str(i), {}).get(f)
                                               for f in ("answer", "work", "file"))]
    if missing:
        st.warning("답안 또는 풀이를 입력해 주세요: " + ", ".join(map(str, missing)) + "번")
        return False
    changed = False
    for i in pending:
        engine = get_engine()
        if not engine:
            break
        question = batch["questions"][i]
        draft = st.session_state.exam_drafts.get(str(i), {})
        try:
            attached = draft.get("file")
            images = solution_file_to_images(MemoryFile(attached["name"], attached["raw"]), max_pages=3) if attached else []
        except Exception as exc:
            st.error(f"{i + 1}번 풀이 파일을 읽지 못했습니다: {exc}")
            continue
        ctx = fit_context(_related_lectures(question["concept"] + " " + question["problem"], k=6)
                          + cited_items(batch["context"], question.get("evidence_ids")))
        with st.spinner(f"{i + 1}번 답안과 풀이를 채점하는 중..."):
            answer = draft.get("answer", "")
            result = call(engine.evaluate_practice_answer, question, answer, draft.get("work", ""),
                          context_to_text(ctx, max_chars=32000), images, _numeric_check(question, answer))
        if not result:
            st.warning(f"{i + 1}번 채점은 완료되지 않았습니다. 작성한 답안은 유지됩니다.")
            continue
        st.session_state.exam_results[str(i)] = result
        changed = True
        st.session_state.exam_grade_contexts[str(i)] = ctx
        if str(i) not in st.session_state.exam_logged:
            verdict = "correct" if result["is_correct"] else ("partial" if result["score"] > 0 else "incorrect")
            record_attempt({"created_at": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
                            "stage": "practice", "concept": question["concept"],
                            "score": result["score"], "verdict": verdict, "error_type": result["error_type"],
                            "question": question["problem"], "student_answer": answer or "(풀이로 제출)",
                            "submitted_work_file": bool(images),
                            "feedback": result["feedback"], "weak_concepts": result["weak_concepts"]})
            st.session_state.exam_logged.append(str(i))
        for concept in result["weak_concepts"]:
            if concept not in st.session_state.weak_queue:
                st.session_state.weak_queue.append(concept)
    return changed


def _feedback(index, question, evidence, batch_id):
    result = st.session_state.exam_results.get(str(index))
    if not result:
        return
    label = "정답" if result["is_correct"] else ("부분 정답" if result["score"] > 0 else "오답")
    st.markdown(f"**{label} · {result['score']}점**")
    st.markdown(result["reason"])
    if result.get("work_assessment"):
        st.markdown("**풀이 과정 평가**")
        st.markdown(result["work_assessment"])
    st.markdown("**피드백**")
    st.markdown(result["feedback"])
    if result["weak_concepts"]:
        st.warning("다시 확인할 개념: " + ", ".join(result["weak_concepts"]))
    evidence(st.session_state.exam_grade_contexts.get(str(index), []), result.get("evidence_ids"),
             title=f"{index + 1}번 채점 근거")
    with st.expander("기준 답안·풀이 확인"):
        st.caption("기준 답안·풀이는 AI가 구성한 학습용 답안입니다. 강의 원문과 함께 확인하세요.")
        st.markdown(question["reference_answer"])
        st.markdown(question["solution"])
    if not result["is_correct"] and st.button("이 개념으로 다시 연습", key=f"exam_retry_{batch_id}_{index}"):
        st.session_state["exam_requested_concept"] = question["concept"]
        st.rerun()


def render_practice(get_engine, call, evidence, refresh, record_attempt):
    st.subheader("문제 풀이")
    st.caption("현재 과목의 강의자료만으로도 문제를 만들 수 있습니다. 기출은 선택 사항입니다.")
    material_upload("exam", refresh, optional=True)
    has_lecture = any(r["kind"] == "lecture" for r in st.session_state.page_records)
    has_exam = any(m["kind"] == "exam" for m in st.session_state.source_files.values())
    if not has_lecture and not has_exam:
        st.info("'자료 설명' 탭에서 강의자료를 올리면 바로 문제를 만들 수 있습니다. 기출문제는 없어도 됩니다.")
        return
    modes = ["강의자료로 문제 만들기"] if has_lecture else []
    if has_exam:
        modes.extend(["기출 원문 풀기", "기출 기반 새 문제"] if has_lecture else ["기출 원문 풀기"])
    mode_key = f"workspace_exam_mode_{scope()}"
    if st.session_state.get(mode_key) not in modes or (has_lecture and "exam_requested_concept" in st.session_state):
        st.session_state[mode_key] = modes[0]
    mode = st.radio("풀이 방식", modes, horizontal=True, key=mode_key)
    original = mode == "기출 원문 풀기"
    lecture_only = mode == "강의자료로 문제 만들기"
    selected, pages, pdf, vision = [], [], None, False
    source_name = "현재 과목 강의자료"
    if not lecture_only:
        meta = _pick_file("exam", "exam")
        records = scoped_records(st.session_state.page_records, "exam", meta["name"])
        count_pages = _page_count(meta, records)
        raw_id = hashlib.sha256(meta["name"].encode() + meta["raw"]).hexdigest()[:12]
        page_text = st.text_input("기출 페이지·슬라이드 범위",
                                 value=f"1-{min(6, count_pages)}" if count_pages > 1 else "1",
                                 key=f"workspace_exam_pages_{scope()}_{raw_id}",
                                 help=f"전체 {count_pages}페이지 중 한 번에 최대 6페이지. 예: 2-5")
        try:
            pages = parse_pages(page_text, count_pages)
        except ValueError as exc:
            st.warning(str(exc))
            return
        selected = scoped_records(records, "exam", pages=pages)
        pdf, source_name = _pdf_for(meta), meta["name"]
        vision = st.checkbox("기출의 그림·수식도 함께 읽기", value=True,
                             key=f"workspace_exam_vision_{scope()}_{raw_id}")
    else:
        st.caption("현재 과목에 등록한 강의자료에서 관련 부분을 찾아 출제합니다. 기출 자료는 사용하지 않습니다.")
    count = int(st.number_input("문제 수", min_value=1, max_value=10, value=5, step=1,
                                key=f"workspace_exam_count_{scope()}"))
    if original:
        st.caption("기출의 수치·조건을 유지해 문제를 추출합니다. 실제 확인한 문항이 부족하면 문제 수나 범위를 조정하도록 안내합니다.")
        if pdf:
            with st.expander("선택한 기출 원본 보기"):
                for page in pages:
                    st.image(render_pdf_page_png(pdf, page), caption=f"{source_name} · p.{page}", width="stretch")
        desired, difficulty, topic, weak = "원문 유지", "medium", "", []
    else:
        x1, x2 = st.columns(2)
        desired = x1.selectbox("문제 유형", ["혼합", "개념 문제", "계산 문제"], key=f"workspace_exam_type_{scope()}")
        difficulty = x2.selectbox("난이도", ["easy", "medium", "hard"], index=1,
                                  format_func=lambda v: {"easy": "쉬움", "medium": "보통", "hard": "어려움"}[v],
                                  key=f"workspace_exam_difficulty_{scope()}")
        if "exam_requested_concept" in st.session_state:
            st.session_state[f"workspace_exam_topic_{scope()}"] = st.session_state.pop("exam_requested_concept")
        topic = st.text_input("연습할 개념·범위 (선택)", key=f"workspace_exam_topic_{scope()}")
        use_weak = st.checkbox("현재 과목의 취약 개념 반영", value=True, key=f"workspace_exam_weak_{scope()}")
        weak = st.session_state.weak_queue[:8] if use_weak else []
        if weak:
            st.caption("현재 과목 취약 개념: " + " · ".join(weak))
    if original and "exam_requested_concept" in st.session_state:
        st.info("강의자료를 등록하면 방금 틀린 개념으로 새 문제를 만들 수 있습니다.")
    blocked = (not original and not has_lecture) or (_needs_vision(selected) and (not pdf or not vision))
    if not has_lecture:
        st.warning("강의자료를 등록하면 수업 범위에 맞춘 해설과 근거 제시가 가능합니다.")
    if _needs_vision(selected) and (not pdf or not vision):
        st.warning("텍스트가 없는 기출은 PDF로 올리고 그림·수식 읽기를 켜 주세요.")
    if st.session_state.exam_set:
        confirm = st.checkbox("현재 풀이를 새 문제 세트로 바꾸기", key=f"workspace_exam_replace_{scope()}")
    else:
        confirm = True
    if st.button(f"{count}문제 시작하기", type="primary", disabled=blocked or not confirm,
                 key=f"workspace_exam_start_{scope()}"):
        engine = get_engine()
        if engine:
            query = topic or " ".join(r["text"][:1000] for r in selected)
            if weak:
                query = " ".join([query, *weak])
            ctx = fit_context(selected + _related_lectures(query, k=12), max_chars=28000)
            readable = [r for r in ctx if not _needs_vision([r])]
            if lecture_only and not readable:
                st.warning("강의자료에서 문제를 만들 텍스트를 읽지 못했습니다. 텍스트가 포함된 PDF·PPTX로 올려 주세요.")
            else:
                with st.spinner(f"{count}문제를 구성하는 중..."):
                    try:
                        result = call(engine.generate_exam_set, context_to_text(ctx, max_chars=36000), ctx,
                                      count, original, desired, difficulty, topic, weak,
                                      _images(pdf, pages, source_name, vision))
                    except Exception as exc:
                        st.error(f"자료를 읽지 못했습니다: {exc}")
                        result = None
                if result:
                    st.session_state.exam_set = {**result, "id": uuid4().hex[:12], "context": ctx,
                                                 "mode": mode, "file_name": source_name, "pages": pages}
                    for key in ["exam_drafts", "exam_results", "exam_grade_contexts"]:
                        st.session_state[key] = {}
                    st.session_state.exam_logged = []
                    st.rerun()
    batch = st.session_state.exam_set
    if not batch:
        return
    questions = batch["questions"]
    batch_id = batch["id"]
    st.divider()
    st.markdown(f"### 현재 풀이 · {len(questions)}문제")
    page_label = f" · {', '.join(map(str, batch['pages']))}페이지" if batch['pages'] else ""
    st.caption(f"{batch['mode']} · {batch['file_name']}{page_label}"
               " · 위에서 바꾼 설정은 새 세트를 시작할 때 적용됩니다.")
    if batch.get("note"):
        st.info(batch["note"])
    display = st.radio("문제 표시", ["전체 문제 보기", "한 문제씩 보기"], horizontal=True,
                       key=f"exam_display_{batch_id}")
    if display == "한 문제씩 보기":
        labels = {i: f"{i + 1}번" + (" · 채점 완료" if str(i) in st.session_state.exam_results else "")
                  for i in range(len(questions))}
        current = st.selectbox("풀 문제", list(range(len(questions))),
                                format_func=labels.get,
                                key=f"exam_current_{batch_id}")
        visible = [current]
    else:
        visible = list(range(len(questions)))
    for i in visible:
        q = questions[i]
        with st.container(border=True):
            st.markdown(f"#### {i + 1}번")
            st.markdown(q["problem"])
            st.caption(f"개념: {q['concept']} · 답 형식: {q['answer_format']}")
            evidence(batch["context"], q.get("evidence_ids"), title=f"{i + 1}번 출제·원문 근거")
            _answer_form(i, q, batch_id)
            if st.button("이 문제 채점", key=f"exam_grade_{batch_id}_{i}", disabled=str(i) in st.session_state.exam_results):
                _grade([i], get_engine, call, record_attempt)
                # Immediate feedback uses the same immutable draft; a rerun locks the widgets.
                if str(i) in st.session_state.exam_results:
                    st.rerun()
            _feedback(i, q, evidence, batch_id)
    if st.button("전체 답안 채점", key=f"exam_grade_all_{batch_id}",
                 disabled=len(st.session_state.exam_results) == len(questions)):
        if _grade(list(range(len(questions))), get_engine, call, record_attempt):
            st.rerun()
    results = st.session_state.exam_results
    if results:
        done, total = len(results), len(questions)
        average = sum(r["score"] for r in results.values()) / done
        st.progress(done / total, text=f"채점 완료 {done}/{total}문제")
        st.metric("채점한 문제 평균", f"{average:.1f}점")
        st.caption("성공한 채점은 자동 저장됩니다. 일부 채점에 실패했다면 나머지 문제만 다시 채점할 수 있습니다.")
