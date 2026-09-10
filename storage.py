from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime


DB_PATH = Path(__file__).with_name("study_ai.db")


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                stage TEXT NOT NULL,
                concept TEXT,
                score INTEGER,
                verdict TEXT,
                error_type TEXT,
                question TEXT,
                student_answer TEXT,
                feedback TEXT,
                weak_concepts TEXT
            )
            """
        )


def log_attempt(
    *,
    stage: str,
    concept: str,
    score: int,
    verdict: str,
    error_type: str,
    question: str,
    student_answer: str,
    feedback: str,
    weak_concepts: list[str],
):
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO attempts (
                created_at, stage, concept, score, verdict, error_type,
                question, student_answer, feedback, weak_concepts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                stage,
                concept,
                int(score),
                verdict,
                error_type,
                question,
                student_answer,
                feedback,
                json.dumps(weak_concepts, ensure_ascii=False),
            ),
        )


def load_attempts(limit: int = 200) -> list[dict]:
    init_db()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM attempts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def reset_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()
