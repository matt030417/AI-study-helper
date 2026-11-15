"""Pure helpers for scoped documents, page ranges, and complete exam sets."""
from __future__ import annotations

import hashlib
import re
from copy import deepcopy


def file_key(kind: str, name: str) -> str:
    return f"{kind}::{name}"


def parse_pages(value: str, page_count: int, limit: int = 6) -> list[int]:
    value = value.strip().replace("–", "-").replace("~", "-")
    if not re.fullmatch(r"\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*", value):
        raise ValueError("페이지는 3 또는 3-5, 8처럼 입력해 주세요.")
    pages = set()
    for part in value.split(","):
        bounds = [int(x.strip()) for x in part.split("-")]
        first, last = bounds[0], bounds[-1]
        if first < 1 or last > page_count or first > last:
            raise ValueError(f"1~{page_count}페이지 안에서 시작과 끝을 확인해 주세요.")
        if last - first + 1 > limit:
            raise ValueError(f"한 번에 최대 {limit}페이지까지 선택할 수 있습니다.")
        pages.update(range(first, last + 1))
        if len(pages) > limit:
            raise ValueError(f"한 번에 최대 {limit}페이지까지 선택할 수 있습니다.")
    return sorted(pages)


def scoped_records(records: list[dict], kind: str, source: str | None = None,
                   pages: list[int] | None = None) -> list[dict]:
    return [dict(r) for r in records if r.get("kind") == kind
            and (source is None or r.get("source") == source)
            and (pages is None or r.get("page") in pages)]


def fit_context(items: list[dict], max_chars: int = 24000) -> list[dict]:
    """Clip BEFORE assigning citation IDs; all visible IDs refer to sent text."""
    result, remaining, seen = [], max_chars, set()
    fair_share = max(200, (max_chars - 120 * len(items)) // max(1, len(items)))
    for item in items:
        identity = (item.get("kind"), item.get("source"), item.get("page"), item.get("text"))
        if identity in seen:
            continue
        seen.add(identity)
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        allowance = min(6000, fair_share, remaining - len(str(item.get("source", ""))) - 100)
        if allowance < 150:
            break
        if len(text) > allowance:
            text = text[:allowance - 30] + "\n[이하 원문 텍스트 일부 생략]"
        result.append({**item, "text": text})
        remaining -= len(result[-1]["text"]) + len(str(item.get("source", ""))) + 100
    return result


def cited_items(items: list[dict], evidence_ids) -> list[dict]:
    ids = []
    for value in evidence_ids or []:
        if type(value) is int and 1 <= value <= len(items) and value not in ids:
            ids.append(value)
    return [items[i - 1] for i in ids]


def validate_exam_set(data: dict, count: int, context: list[dict], original: bool) -> dict:
    data = deepcopy(data)
    questions = data.get("questions") or []
    if len(questions) != count:
        note = data.get("note") or "선택한 범위에서 요청한 수만큼 문제를 구성하지 못했습니다."
        raise ValueError(f"{count}문제를 요청했지만 {len(questions)}문제만 확인됐습니다. {note} 문제 수나 범위를 조정해 주세요.")
    seen = set()
    for q in questions:
        signature = re.sub(r"\W+", "", q.get("problem", "")).casefold()
        if not signature or signature in seen:
            raise ValueError("비어 있거나 중복된 문제가 있어 문제 세트를 열지 않았습니다. 다시 생성해 주세요.")
        seen.add(signature)
        evidence = cited_items(context, q.get("evidence_ids"))
        if not evidence:
            raise ValueError("출처를 확인할 수 없는 문제가 있어 문제 세트를 열지 않았습니다.")
        q["evidence_ids"] = [i for i in q.get("evidence_ids", [])
                             if type(i) is int and 1 <= i <= len(context)]
        if original and not any(item.get("kind") == "exam" for item in evidence):
            raise ValueError("기출 원문 위치를 확인하지 못했습니다. 범위를 좁혀 다시 시도해 주세요.")
        if not original and not any(item.get("kind") == "lecture" for item in evidence):
            raise ValueError("강의자료 근거가 없는 문제가 있어 문제 세트를 열지 않았습니다. 범위나 자료를 확인해 주세요.")
        if not str(q.get("reference_answer", "")).strip() or not str(q.get("solution", "")).strip():
            raise ValueError("기준 답안이 누락돼 문제 세트를 열지 않았습니다.")
    data["questions"] = questions
    return data


def reader_id(user: str, project: str, chapter: str, name: str, raw: bytes,
              pages: list[int], vision: bool) -> str:
    h = hashlib.sha256()
    h.update(raw)
    h.update(repr((user, project, chapter, name, pages, vision)).encode())
    return h.hexdigest()[:24]


def new_workspace_state() -> dict:
    return {"page_records": [], "reader_notes": {}, "reader_chats": {},
            "preview_pdfs": {}, "exam_set": None, "exam_drafts": {},
            "exam_results": {}, "exam_logged": [], "exam_grade_contexts": {}}
