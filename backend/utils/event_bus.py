"""
Event bus for streaming logs and data to the frontend.
Writes every event to an in-memory log so reconnecting WebSocket clients can catch up.
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class EventType(Enum):
    LOG = "log"
    DATA = "data"
    STATUS = "status"
    ERROR = "error"
    PROGRESS = "progress"


@dataclass
class Event:
    event_id: int
    project_id: int
    event_type: EventType
    payload: Any
    timestamp: float
    url: Optional[str] = None


class EventBus:
    """Thread-safe event bus with replay capability for WebSocket reconnection."""

    def __init__(self, project_id: int):
        self.project_id = project_id
        self._events: list[Event] = []
        self._counter = 0
        self._lock = asyncio.Lock()
        self._subscribers: list[Callable[[Event], Coroutine]] = []

    async def emit(self, event_type: EventType, payload: Any, url: str = None):
        async with self._lock:
            self._counter += 1
            event = Event(
                event_id=self._counter,
                project_id=self.project_id,
                event_type=event_type,
                payload=payload,
                timestamp=time.time(),
                url=url,
            )
            self._events.append(event)

        # Broadcast to all live subscribers (non-blocking)
        for callback in self._subscribers:
            try:
                await asyncio.wait_for(callback(event), timeout=5.0)
            except Exception:
                pass  # Don't let a slow subscriber block the pipeline

    async def log(self, message: str, url: str = None):
        print(f"[Project {self.project_id}] {message}")
        await self.emit(EventType.LOG, message, url)

    async def data(self, extracted: dict, url: str):
        await self.emit(EventType.DATA, extracted, url)

    async def error(self, message: str, url: str = None):
        print(f"[Project {self.project_id}] ERROR: {message}")
        await self.emit(EventType.ERROR, message, url)

    async def status(self, status: str):
        await self.emit(EventType.STATUS, status)

    def subscribe(self, callback: Callable[[Event], Coroutine]):
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        self._subscribers = [s for s in self._subscribers if s is not callback]

    def replay_since(self, last_event_id: int = 0) -> list[Event]:
        """Return all events after the given ID for WebSocket reconnection."""
        return [e for e in self._events if e.event_id > last_event_id]
