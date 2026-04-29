from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .memory import SharedMemory
from .models import ProjectSnapshot, TaskResult, TaskSpec
from .project_inspector import build_suggested_commands, infer_metric_targets, inspect_project


class Agent(Protocol):
    role: str

    async def run(
        self,
        task: TaskSpec,
        memory: SharedMemory,
        goal: str,
    ) -> TaskResult:
        ...


def _require_snapshot(memory: SharedMemory) -> ProjectSnapshot:
    scan_entry = memory.latest("scanner")
    if scan_entry is None:
        raise ValueError("Project scan results are missing.")
    snapshot_payload = scan_entry.payload.get("snapshot")
    if not isinstance(snapshot_payload, Mapping):
        raise ValueError("Scanner payload did not contain a project snapshot.")
    return ProjectSnapshot.from_dict(snapshot_payload)


def _latest_payload(memory: SharedMemory, topic: str) -> dict[str, Any]:
    entry = memory.latest(topic)
    return dict(entry.payload) if entry is not None else {}


@dataclass(frozen=True)
class ScannerAgent:
    role: str = "scanner"

    async def run(
        self,
        task: TaskSpec,
        memory: SharedMemory,
        goal: str,
    ) -> TaskResult:
        del memory, goal
        project_root = Path(str(task.metadata.get("project_root", "")))
        snapshot = inspect_project(project_root)
        commands = build_suggested_commands(snapshot)
        metrics = infer_metric_targets(snapshot)
        spawned_tasks = (
            TaskSpec(
                task_id="plan-workflow",
                title="Plan the experiment workflow",
                agent_role="planner",
                instructions="Build the multi-agent experiment plan from the scanned project facts.",
                depends_on=(task.task_id,),
                metadata={"project_root": snapshot.project_root},
            ),
        )
        summary = (
            f"Scanned {snapshot.project_name} and found {len(snapshot.train_entrypoints)} "
            f"training entrypoint(s), {len(snapshot.test_entrypoints)} evaluation entrypoint(s), "
            f"and {len(snapshot.config_files)} config file(s)."
        )
        return TaskResult(
            summary=summary,
            payload={
                "snapshot": snapshot.to_dict(),
                "suggested_commands": commands,
                "metric_targets": list(metrics),
            },
            spawned_tasks=spawned_tasks,
        )


@dataclass(frozen=True)
class PlannerAgent:
    role: str = "planner"

    async def run(
        self,
        task: TaskSpec,
        memory: SharedMemory,
        goal: str,
    ) -> TaskResult:
        del task
        snapshot = _require_snapshot(memory)
        scan_payload = _latest_payload(memory, "scanner")
        commands = dict(scan_payload.get("suggested_commands", {}))
        metrics = tuple(scan_payload.get("metric_targets", infer_metric_targets(snapshot)))
        spawned_tasks = (
            TaskSpec(
                task_id="check-environment",
                title="Check the experiment environment",
                agent_role="environment",
                instructions="Resolve the dependency bootstrap and highlight environment risks.",
                depends_on=("plan-workflow",),
                metadata={"project_root": snapshot.project_root},
            ),
            TaskSpec(
                task_id="prepare-execution-plan",
                title="Prepare the train and evaluation command plan",
                agent_role="executor",
                instructions="Generate train and test commands with validation checkpoints.",
                depends_on=("plan-workflow",),
                metadata={
                    "phase": "plan",
                    "project_root": snapshot.project_root,
                    "commands": commands,
                    "metrics": list(metrics),
                },
            ),
            TaskSpec(
                task_id="simulate-run",
                title="Simulate execution and validation",
                agent_role="executor",
                instructions="Perform a dry-run readiness check for the full experiment path.",
                depends_on=("check-environment", "prepare-execution-plan"),
                metadata={
                    "phase": "validate",
                    "project_root": snapshot.project_root,
                    "commands": commands,
                    "metrics": list(metrics),
                },
            ),
            TaskSpec(
                task_id="diagnose-failures",
                title="Diagnose likely execution failures",
                agent_role="diagnoser",
                instructions="Convert readiness gaps into explicit remediation steps.",
                depends_on=("simulate-run",),
                metadata={"project_root": snapshot.project_root},
            ),
            TaskSpec(
                task_id="publish-report",
                title="Publish the benchmark readiness report",
                agent_role="reporter",
                instructions="Compile the final benchmark, risks, and GitHub handoff summary.",
                depends_on=("plan-workflow", "simulate-run", "diagnose-failures"),
                metadata={"project_root": snapshot.project_root},
            ),
            TaskSpec(
                task_id="approval-gate",
                title="Run the final approval gate",
                agent_role="approver",
                instructions="Check whether the repository is ready for a GitHub handoff.",
                depends_on=("publish-report",),
                requires_approval=True,
                metadata={"project_root": snapshot.project_root},
            ),
        )
        return TaskResult(
            summary="Built a six-stage workflow with parallel environment and execution planning.",
            payload={
                "objective": goal,
                "project_name": snapshot.project_name,
                "parallel_waves": [
                    ["check-environment", "prepare-execution-plan"],
                    ["simulate-run"],
                    ["diagnose-failures"],
                    ["publish-report"],
                    ["approval-gate"],
                ],
                "commands": commands,
                "metric_targets": list(metrics),
            },
            spawned_tasks=spawned_tasks,
        )


@dataclass(frozen=True)
class EnvironmentAgent:
    role: str = "environment"

    async def run(
        self,
        task: TaskSpec,
        memory: SharedMemory,
        goal: str,
    ) -> TaskResult:
        del task, goal
        snapshot = _require_snapshot(memory)
        commands = build_suggested_commands(snapshot)
        setup_command = commands.get("environment")
        warnings: list[str] = []
        if setup_command is None:
            warnings.append("No dependency bootstrap command was inferred.")
        if not snapshot.dependency_files:
            warnings.append("Dependency files are missing from the project root.")
        if not snapshot.config_files:
            warnings.append("No YAML experiment configs were detected.")
        setup_commands = [setup_command] if setup_command else []
        return TaskResult(
            summary="Resolved the environment bootstrap path and highlighted setup risks.",
            payload={
                "setup_commands": setup_commands,
                "dependency_files": list(snapshot.dependency_files),
                "warnings": warnings,
            },
        )


@dataclass(frozen=True)
class ExecutionAgent:
    role: str = "executor"

    async def run(
        self,
        task: TaskSpec,
        memory: SharedMemory,
        goal: str,
    ) -> TaskResult:
        del goal
        snapshot = _require_snapshot(memory)
        phase = str(task.metadata.get("phase", "plan"))
        command_payload = task.metadata.get("commands", {})
        commands = dict(command_payload) if isinstance(command_payload, Mapping) else {}
        metrics = tuple(str(metric) for metric in task.metadata.get("metrics", ()))

        if phase == "plan":
            return TaskResult(
                summary="Prepared train and evaluation commands with validation checkpoints.",
                payload={
                    "status": "planned",
                    "train_command": commands.get("train"),
                    "test_command": commands.get("test"),
                    "metrics": list(metrics),
                    "validation_steps": [
                        "Verify dependency installation succeeds.",
                        "Confirm dataset paths exist before training.",
                        "Run a bounded evaluation command before full training.",
                    ],
                },
            )

        environment_payload = _latest_payload(memory, "environment")
        missing_assets: list[str] = []
        if commands.get("train") is None:
            missing_assets.append("training entrypoint")
        if commands.get("test") is None:
            missing_assets.append("evaluation entrypoint")
        if not snapshot.dataset_hints:
            missing_assets.append("dataset path declaration")
        environment_ready = bool(environment_payload.get("setup_commands"))
        if not environment_ready:
            missing_assets.append("environment bootstrap command")
        status = "ready" if not missing_assets else "needs_attention"
        return TaskResult(
            summary=f"Simulated experiment dry-run and marked the workflow as {status}.",
            payload={
                "status": status,
                "environment_ready": environment_ready,
                "train_command": commands.get("train"),
                "test_command": commands.get("test"),
                "metrics": list(metrics),
                "missing_assets": missing_assets,
                "artifact_targets": [
                    "benchmark_summary.json",
                    "diagnosis.md",
                    "runbook.md",
                ],
            },
        )


@dataclass(frozen=True)
class DiagnoserAgent:
    role: str = "diagnoser"

    async def run(
        self,
        task: TaskSpec,
        memory: SharedMemory,
        goal: str,
    ) -> TaskResult:
        del task, goal
        snapshot = _require_snapshot(memory)
        validation_payload = _latest_payload(memory, "executor")
        issues: list[str] = []
        recommended_fixes: list[str] = []

        if not validation_payload.get("environment_ready", False):
            issues.append("The workflow cannot recreate the Python environment yet.")
            recommended_fixes.append(
                "Add a requirements.txt, env.yaml, or pyproject bootstrap command to the repository root."
            )

        missing_assets = list(validation_payload.get("missing_assets", ()))
        for asset in missing_assets:
            if asset == "training entrypoint":
                issues.append("A training entrypoint is missing.")
                recommended_fixes.append("Expose a train.py, train.sh, or equivalent training entrypoint.")
            elif asset == "evaluation entrypoint":
                issues.append("An evaluation entrypoint is missing.")
                recommended_fixes.append("Expose a test.py, eval.py, or inference script for validation.")
            elif asset == "dataset path declaration":
                issues.append("Dataset paths are not declared in README or YAML config files.")
                recommended_fixes.append("Add explicit dataset root declarations to README or options/*.yml.")
            elif asset == "environment bootstrap command":
                issues.append("The repository does not declare how to bootstrap its environment.")

        if not snapshot.checkpoint_references:
            issues.append("Checkpoint lineage is not documented.")
            recommended_fixes.append("Document resume, checkpoint, or pretrained weight paths in README or YAML files.")

        unique_issues = tuple(dict.fromkeys(issues))
        unique_fixes = tuple(dict.fromkeys(recommended_fixes))
        readiness_score = max(55, 95 - len(unique_issues) * 10)
        recommendation = "ready" if readiness_score >= 80 else "needs_attention"
        return TaskResult(
            summary="Analyzed readiness gaps and generated remediation steps.",
            payload={
                "issues": list(unique_issues),
                "recommended_fixes": list(unique_fixes),
                "readiness_score": readiness_score,
                "recommendation": recommendation,
            },
        )


@dataclass(frozen=True)
class ReporterAgent:
    role: str = "reporter"

    async def run(
        self,
        task: TaskSpec,
        memory: SharedMemory,
        goal: str,
    ) -> TaskResult:
        del task, goal
        snapshot = _require_snapshot(memory)
        environment_payload = _latest_payload(memory, "environment")
        execution_payload = _latest_payload(memory, "executor")
        diagnosis_payload = _latest_payload(memory, "diagnoser")
        planner_payload = _latest_payload(memory, "planner")
        readiness_score = int(diagnosis_payload.get("readiness_score", 0))
        report = {
            "project_name": snapshot.project_name,
            "project_root": snapshot.project_root,
            "detected_assets": {
                "readmes": list(snapshot.readme_files),
                "dependencies": list(snapshot.dependency_files),
                "train_entrypoints": list(snapshot.train_entrypoints),
                "test_entrypoints": list(snapshot.test_entrypoints),
                "config_files": list(snapshot.config_files),
            },
            "commands": {
                "environment": list(environment_payload.get("setup_commands", ())),
                "train": execution_payload.get("train_command"),
                "test": execution_payload.get("test_command"),
            },
            "metrics": list(planner_payload.get("metric_targets", ())),
            "issues": list(diagnosis_payload.get("issues", ())),
            "recommended_fixes": list(diagnosis_payload.get("recommended_fixes", ())),
            "readiness_score": readiness_score,
        }
        return TaskResult(
            summary=(
                f"Generated the benchmark readiness report for {snapshot.project_name} "
                f"with score {readiness_score}/100."
            ),
            payload={
                "report": report,
                "readiness_score": readiness_score,
                "github_handoff": [
                    "README.md",
                    "pyproject.toml",
                    "src/image_research_agent",
                    "tests",
                ],
            },
        )


@dataclass(frozen=True)
class ApprovalAgent:
    role: str = "approver"

    async def run(
        self,
        task: TaskSpec,
        memory: SharedMemory,
        goal: str,
    ) -> TaskResult:
        del task, goal
        report_payload = _latest_payload(memory, "reporter")
        readiness_score = int(report_payload.get("readiness_score", 0))
        approval = readiness_score >= 80
        guardrails = [
            "Keep dataset paths configurable and out of source control.",
            "Require human approval before enabling destructive tool adapters.",
            "Persist benchmark artifacts for reproducibility and audit.",
        ]
        return TaskResult(
            summary="Completed the final approval gate for repository handoff.",
            payload={
                "approval": approval,
                "readiness_score": readiness_score,
                "guardrails": guardrails,
            },
        )


@dataclass(frozen=True)
class AgentRegistry:
    agents: dict[str, Agent]

    def get(self, role: str) -> Agent:
        if role not in self.agents:
            raise ValueError(f"Unknown agent role '{role}'.")
        return self.agents[role]

    @classmethod
    def default(cls) -> "AgentRegistry":
        built_in_agents: tuple[Agent, ...] = (
            ScannerAgent(),
            PlannerAgent(),
            EnvironmentAgent(),
            ExecutionAgent(),
            DiagnoserAgent(),
            ReporterAgent(),
            ApprovalAgent(),
        )
        return cls(agents={agent.role: agent for agent in built_in_agents})
