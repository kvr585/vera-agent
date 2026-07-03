"""SQLite adapter for storing agent session states."""

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from vera_engine.core.entities import AgentState
from vera_engine.core.interfaces.repository import SessionRepository


class SQLiteSessionRepository(SessionRepository):
    """SQLite implementation of SessionRepository, persisting JSON snapshots."""

    def __init__(self, db_path: str = "logs/vera.db") -> None:
        """Initializes database schema.

        Args:
            db_path: Path to database file.
        """
        clean_path = db_path
        if clean_path.startswith("sqlite:///"):
            clean_path = clean_path[len("sqlite:///") :]
        path = Path(clean_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_state(self, state: AgentState) -> None:
        """Persists agent state into sqlite table."""
        state_json = state.model_dump_json()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(session_id, state_json) VALUES (?, ?)",
                (state.session_id, state_json),
            )
            conn.commit()

    def get_state(self, session_id: str) -> AgentState | None:
        """Retrieves and deserializes session state from sqlite table."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                return AgentState.model_validate_json(row[0])
        return None

    def list_sessions(self) -> Sequence[AgentState]:
        """Lists all stored session states."""
        states = []
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT state_json FROM sessions")
            for row in cursor.fetchall():
                states.append(AgentState.model_validate_json(row[0]))
        return states
