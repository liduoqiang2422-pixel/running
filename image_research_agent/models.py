from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from .memory import MemoryEntry, SharedMemory


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ProjectSnapshot:
    project_name: str
    project_root: str
    readme_files: tuple[str, ...] = ()
    dependency_files: tuple[str, ...] = ()
    train_entrypoints: tuple[str, ...] = ()
    test_entrypoints: tuple[str, ...] = ()
    config_files: tuple[str, ...] = ()
    checkpoint_references: tuple[str, ...] = ()
    dataset_hints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "project_root": self.project_root,
            "readme_files": list(self.readme_files),
            "dependency_files": list(self.dependency_files),
            "train_entrypoints": list(self.train_entrypoints),
            "test_entrypoints": list(self.test_entrypoints),
            "config_files": list(self.config_files),
            "checkpoint_references": list(self.checkpoint_references),
            "dataset_hints": list(self.dataset_hints),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectSnapshot":
        return cls(
            project_name=str(payload["project_name"]),
            project_root=str(payload["project_root"]),
            readme_files=tuple(str(item) for item in payload.get("readme_files", ())),
            dependency_files=tuple(
                str(item) for item in payload.get("dependency_files", ())
            ),
            train_entrypoints=tuple(
                str(item) for item in payload.get("train_entrypoints", ())
            ),
            test_entrypoints=tuple(
                str(item) for item in payload.get("test_entrypoints", ())
            ),
            config_files=tuple(str(item) for item in payload.get("config_files", ())),
            checkpoint_references=tuple(
                str(item) for item in payload.get("checkpoint_references", ())
            ),
            dataset_hints=tuple(str(item) for item in payload.get("dataset_hints", ())),
            tags=tuple(str(item) for item in payload.get("tags", ())),
        )


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    agent_role: str
    instructions: str
    depends_on: tuple[str, ...] = ()
    requires_approval: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    summary: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    spawned_tasks: tuple[TaskSpec, ...] = ()


@dataclass(frozen=True)
class Artifact:
    task_id: str
    title: str
    agent_role: str
    summary: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionState:
    workflow_id: str
    goal: str
    project_root: str
    tasks: Mapping[str, TaskSpec] = field(default_factory=dict)
    statuses: Mapping[str, TaskStatus] = field(default_factory=dict)
    artifacts: tuple[Artifact, ...] = ()
    memory: SharedMemory = field(default_factory=SharedMemory)
    history: tuple[str, ...] = ()
    waiting_approval: tuple[str, ...] = ()
    approved_tasks: tuple[str, ...] = ()

    def add_tasks(self, new_tasks: Sequence[TaskSpec]) -> "ExecutionState":
        if not new_tasks:
            return self
        updated_tasks = dict(self.tasks)
        updated_statuses = dict(self.statuses)
        updated_history = list(self.history)
        for task in new_tasks:
            if task.task_id in updated_tasks:
                raise ValueError(f"Task '{task.task_id}' already exists.")
            updated_tasks[task.task_id] = task
            updated_statuses[task.task_id] = TaskStatus.PENDING
            updated_history.append(f"registered:{task.task_id}")
        return replace(
            self,
            tasks=updated_tasks,
            statuses=updated_statuses,
            history=tuple(updated_history),
        )

    def update_status(self, task_id: str, status: TaskStatus) -> "ExecutionState":
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task '{task_id}'.")
        updated_statuses = dict(self.statuses)
        updated_statuses[task_id] = status
        return replace(self, statuses=updated_statuses)

    def add_artifact(self, artifact: Artifact) -> "ExecutionState":
        return replace(self, artifacts=(*self.artifacts, artifact))

    def add_memory_entry(self, entry: MemoryEntry) -> "ExecutionState":
        return replace(self, memory=self.memory.append(entry))

    def add_history(self, event: str) -> "ExecutionState":
        return replace(self, history=(*self.history, event))

    def mark_waiting_approval(self, task_id: str) -> "ExecutionState":
        if task_id in self.waiting_approval:
            return self
        updated_state = self.update_status(task_id, TaskStatus.WAITING_APPROVAL)
        return replace(
            updated_state,
            waiting_approval=(*updated_state.waiting_approval, task_id),
        )

    def approve_tasks(self, task_ids: Sequence[str]) -> "ExecutionState":
        approved_ids = set(self.approved_tasks)
        waiting_ids = list(self.waiting_approval)
        updated_state = self
        for task_id in task_ids:
            if task_id not in self.tasks:
                raise ValueError(f"Unknown task '{task_id}'.")
            approved_ids.add(task_id)
            waiting_ids = [candidate for candidate in waiting_ids if candidate != task_id]
            if self.statuses.get(task_id) == TaskStatus.WAITING_APPROVAL:
                updated_state = updated_state.update_status(task_id, TaskStatus.PENDING)
        return replace(
            updated_state,
            approved_tasks=tuple(sorted(approved_ids)),
            waiting_approval=tuple(waiting_ids),
        )


@dataclass(frozen=True)
class ExecutionReport:
    state: ExecutionState

    @property
    def pending_approvals(self) -> tuple[str, ...]:
        return self.state.waiting_approval

    @property
    def is_complete(self) -> bool:
        return bool(self.state.tasks) and all(
            status == TaskStatus.COMPLETED
            for status in self.state.statuses.values()
        )

    def to_dict(self) -> dict[str, Any]:
        tasks = {
            task_id: {
                "title": task.title,
                "agent_role": task.agent_role,
                "depends_on": list(task.depends_on),
                "requires_approval": task.requires_approval,
                "status": self.state.statuses[task_id].value,
                "metadata": dict(task.metadata),
            }
            for task_id, task in self.state.tasks.items()
        }
        artifacts = [
            {
                "task_id": artifact.task_id,
                "title": artifact.title,
                "agent_role": artifact.agent_role,
                "summary": artifact.summary,
                "payload": dict(artifact.payload),
            }
            for artifact in self.state.artifacts
        ]
        memory = [
            {
                "topic": entry.topic,
                "source_task": entry.source_task,
                "content": entry.content,
                "payload": dict(entry.payload),
            }
            for entry in self.state.memory.entries
        ]
        return {
            "workflow_id": self.state.workflow_id,
            "goal": self.state.goal,
            "project_root": self.state.project_root,
            "is_complete": self.is_complete,
            "pending_approvals": list(self.pending_approvals),
            "approved_tasks": list(self.state.approved_tasks),
            "tasks": tasks,
            "artifacts": artifacts,
            "memory": memory,
            "history": list(self.state.history),
        }
