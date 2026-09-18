"""Concurrency-safe, process-local session state for the initial deployment."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock


@dataclass
class ProctorSessionState:
    owner_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latest_result: dict = field(default_factory=dict)
    last_frame_at: datetime | None = None
    errors: list[str] = field(default_factory=list)


class SessionStore:
    def __init__(self):
        self._lock = RLock()
        self._sessions = {}

    def start(self, session_id, owner_id):
        with self._lock:
            state = ProctorSessionState(owner_id=owner_id)
            self._sessions[session_id] = state
            return state

    def get_owned(self, session_id, owner_id):
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.owner_id != owner_id:
                return None
            return state

    def stop(self, session_id, owner_id):
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.owner_id != owner_id:
                return False
            del self._sessions[session_id]
            return True


sessions = SessionStore()
