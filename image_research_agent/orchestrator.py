from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Iterable

from .agents import AgentRegistry
from .memory import MemoryEntry
from .models import Artifact, ExecutionReport, ExecutionState, TaskResult, TaskSpec, TaskStatus


def _slugify(value: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_length] or "workflow"


class ResearchExperimentOrchestrator:
    def __init__(
        self,
        registry: AgentRegistry | None = None,
        max_concurrency: int = 3,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        self._registry = registry or AgentRegistry.default()
        self._max_concurrency = max_concurrency

    def create_workflow(self, project_root: Path | str, goal: str) -> ExecutionState:
        normalized_goal = goal.strip()
        resolved_root = Path(project_root).resolve()
        if not normalized_goal:
            raise ValueError("Goal must not be empty.")
        if not resolved_root.is_dir():
            raise ValueError(f"Project path does not exist or is not a directory: {project_root}")
        scanning_task = TaskSpec(
            task_id="scan-project",
            title="Scan the research project",
            agent_role="scanner",
            instructions="Inspect the project structure and extract experiment metadata.",
            metadata={"project_root": str(resolved_root)},
        )
        state = ExecutionState(
            workflow_id=f"exp-{_slugify(normalized_goal)}-{_slugify(resolved_root.name, 16)}",
            goal=normalized_goal,
            project_root=str(resolved_root),
        )
        return state.add_tasks((scanning_task,))

    async def run_project(
        self,
        project_root: Path | str,
        goal: str,
        auto_approve: bool = False,
    ) -> ExecutionReport:
        initial_state = self.create_workflow(project_root=project_root, goal=goal)
        return await self.run(initial_state, auto_approve=auto_approve)

    async def run(
        self,
        state: ExecutionState,
        auto_approve: bool = False,
    ) -> ExecutionReport:
        current_state = state
        while True:
            ready_tasks = self._ready_tasks(current_state)
            if not ready_tasks:
                return ExecutionReport(state=current_state)
            current_state, progressed = await self._execute_wave(
                current_state,
                ready_tasks,
                auto_approve=auto_approve,
            )
            if not progressed:
                return ExecutionReport(state=current_state)

    def _ready_tasks(self, state: ExecutionState) -> list[TaskSpec]:
        ready_tasks: list[TaskSpec] = []
        for task_id, task in state.tasks.items():
            if state.statuses.get(task_id) != TaskStatus.PENDING:
                continue
            dependency_states = [state.statuses.get(depends_on) for depends_on in task.depends_on]
            if all(status == TaskStatus.COMPLETED for status in dependency_states):
                ready_tasks.append(task)
        return sorted(ready_tasks, key=lambda task: task.task_id)

    async def _execute_wave(
        self,
        state: ExecutionState,
        ready_tasks: Iterable[TaskSpec],
        auto_approve: bool,
    ) -> tuple[ExecutionState, bool]:
        current_state = state
        runnable_tasks: list[TaskSpec] = []
        approvals_recorded = False
        for task in ready_tasks:
            if task.requires_approval and task.task_id not in current_state.approved_tasks:
                if auto_approve:
                    current_state = current_state.approve_tasks((task.task_id,))
                else:
                    current_state = current_state.mark_waiting_approval(task.task_id)
                    current_state = current_state.add_history(f"approval_required:{task.task_id}")
                    approvals_recorded = True
                    continue
            runnable_tasks.append(task)
        if not runnable_tasks:
            return current_state, approvals_recorded

        for task in runnable_tasks:
            current_state = current_state.update_status(task.task_id, TaskStatus.RUNNING)
            current_state = current_state.add_history(f"running:{task.task_id}")

        task_results = await self._invoke_tasks(runnable_tasks, current_state)
        for task, result, error_message in task_results:
            if error_message is not None:
                current_state = current_state.update_status(task.task_id, TaskStatus.FAILED)
                current_state = current_state.add_history(
                    f"failed:{task.task_id}:{error_message}"
                )
                continue

            artifact = Artifact(
                task_id=task.task_id,
                title=task.title,
                agent_role=task.agent_role,
                summary=result.summary,
                payload=result.payload,
            )
            current_state = current_state.update_status(task.task_id, TaskStatus.COMPLETED)
            current_state = current_state.add_artifact(artifact)
            current_state = current_state.add_memory_entry(
                MemoryEntry(
                    topic=task.agent_role,
                    source_task=task.task_id,
                    content=result.summary,
                    payload=result.payload,
                )
            )
            current_state = current_state.add_history(f"completed:{task.task_id}")
            if result.spawned_tasks:
                current_state = current_state.add_tasks(result.spawned_tasks)

        return current_state, True

    async def _invoke_tasks(
        self,
        tasks: list[TaskSpec],
        state: ExecutionState,
    ) -> list[tuple[TaskSpec, TaskResult | None, str | None]]:
        semaphore = asyncio.Semaphore(self._max_concurrency)
        memory_snapshot = state.memory

        async def invoke(
            task: TaskSpec,
        ) -> tuple[TaskSpec, TaskResult | None, str | None]:
            agent = self._registry.get(task.agent_role)
            async with semaphore:
                try:
                    result = await agent.run(task=task, memory=memory_snapshot, goal=state.goal)
                except Exception as exc:
                    return task, None, str(exc)
            return task, result, None

        return list(await asyncio.gather(*(invoke(task) for task in tasks)))
