from __future__ import annotations

from typing import Any

from supabase import Client, create_client


USERNAME_DOMAIN = "study-helper.invalid"


def normalize_username(username: str) -> str:
    value = username.strip().lower()
    # Simple IDs only: letters, numbers, underscore, hyphen, dot.
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    if not value or any(ch not in allowed for ch in value):
        raise ValueError("학습 ID는 영문 소문자, 숫자, ., _, - 만 사용할 수 있습니다.")
    if len(value) < 3 or len(value) > 30:
        raise ValueError("학습 ID는 3~30자로 만들어 주세요.")
    return value


def username_to_internal_email(username: str) -> str:
    return f"{normalize_username(username)}@{USERNAME_DOMAIN}"



def new_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)


def sign_in(url: str, key: str, username: str, password: str):
    client = new_supabase_client(url, key)
    internal_email = username_to_internal_email(username)
    response = client.auth.sign_in_with_password(
        {"email": internal_email, "password": password}
    )
    return client, response


def sign_up(url: str, key: str, username: str, password: str):
    client = new_supabase_client(url, key)
    normalized = normalize_username(username)
    internal_email = username_to_internal_email(normalized)
    response = client.auth.sign_up(
        {
            "email": internal_email,
            "password": password,
            "options": {"data": {"username": normalized}},
        }
    )
    return client, response


def sign_out(client: Client):
    client.auth.sign_out()


def _normalize_attempt(row: dict[str, Any]) -> dict[str, Any]:
    weak = row.get("weak_concepts") or []
    if not isinstance(weak, list):
        weak = []

    created = str(row.get("created_at") or "")
    if "T" in created:
        created = created.replace("T", " ")[:19]

    return {
        "id": row.get("id"),
        "created_at": created,
        "stage": row.get("stage") or "",
        "concept": row.get("concept") or "",
        "score": int(row.get("score") or 0),
        "verdict": row.get("verdict") or "",
        "error_type": row.get("error_type") or "none",
        "question": row.get("question") or "",
        "student_answer": row.get("student_answer") or "",
        "submitted_work_file": bool(row.get("submitted_work_file", False)),
        "feedback": row.get("feedback") or "",
        "weak_concepts": weak,
    }


def load_attempts(client: Client, user_id: str, limit: int = 500) -> list[dict]:
    response = (
        client.table("learning_attempts")
        .select(
            "id,created_at,stage,concept,score,verdict,error_type,"
            "question,student_answer,submitted_work_file,feedback,weak_concepts"
        )
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return [_normalize_attempt(row) for row in (response.data or [])]


def save_attempt(client: Client, user_id: str, attempt: dict) -> dict:
    payload = {
        "user_id": user_id,
        "stage": attempt.get("stage") or "",
        "concept": attempt.get("concept") or "",
        "score": int(attempt.get("score") or 0),
        "verdict": attempt.get("verdict") or "",
        "error_type": attempt.get("error_type") or "none",
        "question": attempt.get("question") or "",
        "student_answer": attempt.get("student_answer") or "",
        "submitted_work_file": bool(attempt.get("submitted_work_file", False)),
        "feedback": attempt.get("feedback") or "",
        "weak_concepts": attempt.get("weak_concepts") or [],
    }

    response = (
        client.table("learning_attempts")
        .insert(payload)
        .select(
            "id,created_at,stage,concept,score,verdict,error_type,"
            "question,student_answer,submitted_work_file,feedback,weak_concepts"
        )
        .execute()
    )
    if response.data:
        return _normalize_attempt(response.data[0])

    # Should rarely happen; return a local-format record as a fallback.
    return _normalize_attempt(payload)


def delete_all_attempts(client: Client, user_id: str):
    return (
        client.table("learning_attempts")
        .delete()
        .eq("user_id", user_id)
        .execute()
    )
