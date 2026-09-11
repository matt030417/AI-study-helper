from __future__ import annotations

from typing import Any
from uuid import uuid4
import mimetypes

from supabase import Client, create_client


USERNAME_DOMAIN = "study-helper.invalid"
STORAGE_BUCKET = "study-files"


def normalize_username(username: str) -> str:
    value = username.strip().lower()
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
    response = client.auth.sign_in_with_password(
        {"email": username_to_internal_email(username), "password": password}
    )
    return client, response


def sign_up(url: str, key: str, username: str, password: str):
    client = new_supabase_client(url, key)
    normalized = normalize_username(username)
    response = client.auth.sign_up(
        {
            "email": username_to_internal_email(normalized),
            "password": password,
            "options": {"data": {"username": normalized}},
        }
    )
    return client, response


def sign_out(client: Client):
    client.auth.sign_out()


def _normalize_project(row: dict[str, Any]) -> dict[str, Any]:
    created = str(row.get("created_at") or "")
    if "T" in created:
        created = created.replace("T", " ")[:19]
    return {
        "id": str(row.get("id") or ""),
        "name": row.get("name") or "프로젝트",
        "description": row.get("description") or "",
        "created_at": created,
    }


def load_projects(client: Client, user_id: str) -> list[dict]:
    response = (
        client.table("study_projects")
        .select("id,name,description,created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .execute()
    )
    return [_normalize_project(x) for x in (response.data or [])]


def create_project(client: Client, user_id: str, name: str, description: str = "") -> dict:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("프로젝트 이름을 입력해 주세요.")
    if len(clean_name) > 80:
        raise ValueError("프로젝트 이름은 80자 이하로 입력해 주세요.")
    response = (
        client.table("study_projects")
        .insert({"user_id": user_id, "name": clean_name, "description": description.strip()})
        .select("id,name,description,created_at")
        .execute()
    )
    if not response.data:
        raise RuntimeError("프로젝트 생성 결과를 받지 못했습니다.")
    return _normalize_project(response.data[0])


def rename_project(client: Client, user_id: str, project_id: str, name: str, description: str = "") -> dict:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("프로젝트 이름을 입력해 주세요.")
    response = (
        client.table("study_projects")
        .update({"name": clean_name, "description": description.strip()})
        .eq("id", project_id)
        .eq("user_id", user_id)
        .select("id,name,description,created_at")
        .execute()
    )
    if not response.data:
        raise RuntimeError("프로젝트 수정 결과를 받지 못했습니다.")
    return _normalize_project(response.data[0])


def _normalize_file(row: dict[str, Any]) -> dict[str, Any]:
    created = str(row.get("created_at") or "")
    if "T" in created:
        created = created.replace("T", " ")[:19]
    return {
        "id": int(row.get("id")) if row.get("id") is not None else None,
        "project_id": str(row.get("project_id") or ""),
        "file_name": row.get("file_name") or "",
        "kind": row.get("kind") or "lecture",
        "suffix": row.get("suffix") or "",
        "storage_path": row.get("storage_path") or "",
        "size_bytes": int(row.get("size_bytes") or 0),
        "created_at": created,
    }


def load_project_files(client: Client, user_id: str, project_id: str) -> list[dict]:
    response = (
        client.table("project_files")
        .select("id,project_id,file_name,kind,suffix,storage_path,size_bytes,created_at")
        .eq("user_id", user_id)
        .eq("project_id", project_id)
        .order("created_at", desc=False)
        .execute()
    )
    return [_normalize_file(x) for x in (response.data or [])]


def download_project_file(client: Client, storage_path: str) -> bytes:
    data = client.storage.from_(STORAGE_BUCKET).download(storage_path)
    return bytes(data)


def save_project_file(
    client: Client,
    user_id: str,
    project_id: str,
    file_name: str,
    kind: str,
    raw: bytes,
) -> dict:
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    existing = (
        client.table("project_files")
        .select("id,storage_path")
        .eq("user_id", user_id)
        .eq("project_id", project_id)
        .eq("file_name", file_name)
        .eq("kind", kind)
        .execute()
    )
    for row in existing.data or []:
        path = row.get("storage_path")
        if path:
            try:
                client.storage.from_(STORAGE_BUCKET).remove([path])
            except Exception:
                pass
        client.table("project_files").delete().eq("id", row["id"]).eq("user_id", user_id).execute()

    safe_name = file_name.replace("/", "_").replace("\\", "_")
    storage_path = f"{user_id}/{project_id}/{uuid4().hex}_{safe_name}"
    content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    client.storage.from_(STORAGE_BUCKET).upload(
        path=storage_path,
        file=raw,
        file_options={"content-type": content_type, "upsert": "false"},
    )
    try:
        response = (
            client.table("project_files")
            .insert(
                {
                    "user_id": user_id,
                    "project_id": project_id,
                    "file_name": file_name,
                    "kind": kind,
                    "suffix": suffix,
                    "storage_path": storage_path,
                    "size_bytes": len(raw),
                }
            )
            .select("id,project_id,file_name,kind,suffix,storage_path,size_bytes,created_at")
            .execute()
        )
    except Exception:
        try:
            client.storage.from_(STORAGE_BUCKET).remove([storage_path])
        except Exception:
            pass
        raise
    if not response.data:
        raise RuntimeError("파일 메타데이터 저장 결과를 받지 못했습니다.")
    return _normalize_file(response.data[0])


def delete_project_file(client: Client, user_id: str, file_id: int):
    response = (
        client.table("project_files")
        .select("id,storage_path")
        .eq("id", file_id)
        .eq("user_id", user_id)
        .execute()
    )
    if response.data:
        path = response.data[0].get("storage_path")
        if path:
            client.storage.from_(STORAGE_BUCKET).remove([path])
    client.table("project_files").delete().eq("id", file_id).eq("user_id", user_id).execute()


def delete_project(client: Client, user_id: str, project_id: str):
    files = load_project_files(client, user_id, project_id)
    paths = [x["storage_path"] for x in files if x.get("storage_path")]
    if paths:
        client.storage.from_(STORAGE_BUCKET).remove(paths)
    return (
        client.table("study_projects")
        .delete()
        .eq("id", project_id)
        .eq("user_id", user_id)
        .execute()
    )


def _normalize_attempt(row: dict[str, Any]) -> dict[str, Any]:
    weak = row.get("weak_concepts") or []
    if not isinstance(weak, list):
        weak = []
    created = str(row.get("created_at") or "")
    if "T" in created:
        created = created.replace("T", " ")[:19]
    return {
        "id": row.get("id"),
        "project_id": str(row.get("project_id") or ""),
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


def load_attempts(client: Client, user_id: str, project_id: str | None = None, limit: int = 1000) -> list[dict]:
    query = (
        client.table("learning_attempts")
        .select(
            "id,project_id,created_at,stage,concept,score,verdict,error_type,"
            "question,student_answer,submitted_work_file,feedback,weak_concepts"
        )
        .eq("user_id", user_id)
    )
    if project_id:
        query = query.eq("project_id", project_id)
    response = query.order("created_at", desc=False).limit(limit).execute()
    return [_normalize_attempt(row) for row in (response.data or [])]


def save_attempt(client: Client, user_id: str, project_id: str, attempt: dict) -> dict:
    if not project_id:
        raise ValueError("현재 프로젝트가 선택되지 않았습니다.")
    payload = {
        "user_id": user_id,
        "project_id": project_id,
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
            "id,project_id,created_at,stage,concept,score,verdict,error_type,"
            "question,student_answer,submitted_work_file,feedback,weak_concepts"
        )
        .execute()
    )
    if response.data:
        return _normalize_attempt(response.data[0])
    return _normalize_attempt(payload)


def delete_project_attempts(client: Client, user_id: str, project_id: str):
    return (
        client.table("learning_attempts")
        .delete()
        .eq("user_id", user_id)
        .eq("project_id", project_id)
        .execute()
    )
