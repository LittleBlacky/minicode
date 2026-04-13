"""Checkpoint management for session persistence."""
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


class CheckpointManager:
    """Manage LangGraph checkpoints for session persistence.

    Supports two backends:
    - MemorySaver: Fast, session-only (loses state on restart)
    - SqliteSaver: Persistent, survives restarts
    """

    def __init__(self, use_sqlite: bool = False, db_path: Optional[str] = None):
        self.use_sqlite = use_sqlite
        self.db_path = db_path
        self._checkpointer = None

    def get_checkpointer(self):
        """Get or create the checkpointer instance."""
        if self._checkpointer is None:
            if self.use_sqlite and self.db_path:
                self._checkpointer = SqliteSaver.from_conn_string(self.db_path)
            else:
                self._checkpointer = MemorySaver()
        return self._checkpointer

    def get_session_config(self, thread_id: str) -> dict:
        """Get LangGraph config for a session."""
        return {"configurable": {"thread_id": thread_id}}

    def clear_session(self, thread_id: str) -> None:
        """Clear a specific session's checkpoint."""
        if hasattr(self._checkpointer, 'delete'):
            config = self.get_session_config(thread_id)
            self._checkpointer.delete(config)


def create_checkpointer(
    use_sqlite: bool = False,
    db_path: Optional[str] = None,
) -> MemorySaver | SqliteSaver:
    """Factory function to create a checkpointer."""
    if use_sqlite and db_path:
        return SqliteSaver.from_conn_string(db_path)
    return MemorySaver()
