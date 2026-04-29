from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MemoryEntry:
    topic: str
    source_task: str
    content: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SharedMemory:
    entries: tuple[MemoryEntry, ...] = ()

    def append(self, entry: MemoryEntry) -> "SharedMemory":
        return SharedMemory(entries=(*self.entries, entry))

    def extend(self, entries: Sequence[MemoryEntry]) -> "SharedMemory":
        return SharedMemory(entries=self.entries + tuple(entries))

    def latest(self, topic: str) -> MemoryEntry | None:
        for entry in reversed(self.entries):
            if entry.topic == topic:
                return entry
        return None

    def to_brief(self, topics: Sequence[str] | None = None) -> str:
        allowed_topics = set(topics) if topics is not None else None
        selected_entries = [
            entry
            for entry in self.entries
            if allowed_topics is None or entry.topic in allowed_topics
        ]
        return "\n".join(
            f"[{entry.topic}] {entry.content}" for entry in selected_entries[-6:]
        )
