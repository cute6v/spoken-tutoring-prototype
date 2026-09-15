"""Layer 6 - Data Management: SQLite-backed store for learner state.

Uses only Python's built-in sqlite3. State persists across turns AND across
process runs (sessions), which is what lets adaptation accumulate over time.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from .domain import CATEGORIES, AssessmentResult, Prompt, ProfileState, Transcript
from .interfaces import IDataStore


class SQLiteDataStore(IDataStore):
    def __init__(self, db_path: str = "tutor_state.db") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                learner_id        TEXT PRIMARY KEY,
                cumulative_errors TEXT NOT NULL,
                total_attempts    INTEGER NOT NULL,
                weakest_category  TEXT,
                last_score        REAL,
                updated_at        TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id    TEXT NOT NULL,
                turn_no       INTEGER NOT NULL,
                prompt_id     TEXT,
                prompt_text   TEXT,
                transcript    TEXT,
                source_mode   TEXT,
                per_category  TEXT,
                score         REAL,
                created_at    TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def load_profile(self, learner_id: str) -> Optional[ProfileState]:
        row = self._conn.execute(
            "SELECT * FROM profiles WHERE learner_id = ?", (learner_id,)
        ).fetchone()
        if row is None:
            return None
        errors = json.loads(row["cumulative_errors"])
        # Make sure every known category is present.
        errors = {c: int(errors.get(c, 0)) for c in CATEGORIES}
        return ProfileState(
            learner_id=row["learner_id"],
            cumulative_errors=errors,
            total_attempts=row["total_attempts"],
            weakest_category=row["weakest_category"],
            last_score=row["last_score"],
        )

    def save_profile(self, profile: ProfileState) -> None:
        self._conn.execute(
            """
            INSERT INTO profiles
                (learner_id, cumulative_errors, total_attempts,
                 weakest_category, last_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(learner_id) DO UPDATE SET
                cumulative_errors = excluded.cumulative_errors,
                total_attempts    = excluded.total_attempts,
                weakest_category  = excluded.weakest_category,
                last_score        = excluded.last_score,
                updated_at        = excluded.updated_at
            """,
            (
                profile.learner_id,
                json.dumps(profile.cumulative_errors),
                profile.total_attempts,
                profile.weakest_category,
                profile.last_score,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def reset_learner(self, learner_id: str) -> None:
        self._conn.execute("DELETE FROM profiles WHERE learner_id = ?", (learner_id,))
        self._conn.execute("DELETE FROM turns WHERE learner_id = ?", (learner_id,))
        self._conn.commit()

    def log_turn(self, learner_id: str, turn: int,
                 prompt: Prompt, transcript: Transcript,
                 result: AssessmentResult) -> None:
        self._conn.execute(
            """
            INSERT INTO turns
                (learner_id, turn_no, prompt_id, prompt_text, transcript,
                 source_mode, per_category, score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                learner_id, turn, prompt.prompt_id, prompt.text,
                transcript.text, transcript.source_mode,
                json.dumps(result.per_category), result.score,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
