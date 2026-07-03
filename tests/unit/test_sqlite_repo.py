"""Unit tests for the SQLite session repository."""

from pathlib import Path

from vera_engine.adapters.repositories.sqlite import SQLiteSessionRepository
from vera_engine.core.entities import AgentState


def test_sqlite_repository_lifecycle(tmp_path: Path) -> None:
    """Verifies SQLiteSessionRepository can save, retrieve, and list sessions."""
    db_file = tmp_path / "test_vera.db"
    SQLiteSessionRepository(db_path=str(db_file))

    # Re-run constructor to test schema creation idempotency (IF NOT EXISTS)
    repo_double = SQLiteSessionRepository(db_path=str(db_file))

    # Retrieve missing session
    assert repo_double.get_state("non_existent") is None
    assert len(repo_double.list_sessions()) == 0

    # Save a test session
    state = AgentState(
        session_id="session-123",
        goal="Test SQLite save",
        current_workflow="default",
    )
    repo_double.save_state(state)

    # Get state
    retrieved = repo_double.get_state("session-123")
    assert retrieved is not None
    assert retrieved.session_id == "session-123"
    assert retrieved.goal == "Test SQLite save"

    # List states
    sessions = repo_double.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_id == "session-123"
