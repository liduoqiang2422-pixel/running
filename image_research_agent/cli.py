from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

from .orchestrator import ResearchExperimentOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="irea",
        description="Run the image research experiment orchestration workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a workflow from direct arguments or a JSON file.")
    goal_source = run_parser.add_mutually_exclusive_group(required=True)
    goal_source.add_argument("--goal", help="Plain-text goal for the workflow.")
    goal_source.add_argument(
        "--from-file",
        help="Path to a JSON file containing 'goal' and 'project_path' keys.",
    )
    run_parser.add_argument(
        "--project-path",
        help="Path to the research project directory. Required when --goal is used.",
    )
    run_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve tasks that are behind an approval gate.",
    )
    run_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum number of tasks to execute concurrently.",
    )
    return parser


def _load_request(args: argparse.Namespace) -> tuple[Path, str, bool]:
    if args.goal:
        if not args.project_path:
            raise ValueError("--project-path is required when --goal is used.")
        return Path(str(args.project_path)), str(args.goal), bool(args.auto_approve)

    payload = json.loads(Path(str(args.from_file)).read_text(encoding="utf-8"))
    project_path = args.project_path or payload.get("project_path")
    if project_path is None:
        raise ValueError("The input file must contain a 'project_path' key or pass --project-path.")
    if "goal" not in payload:
        raise ValueError("The input file must contain a 'goal' key.")
    file_auto_approve = bool(payload.get("auto_approve", False))
    return Path(str(project_path)), str(payload["goal"]), bool(args.auto_approve or file_auto_approve)


async def _run_command(args: argparse.Namespace) -> int:
    project_path, goal, auto_approve = _load_request(args)
    orchestrator = ResearchExperimentOrchestrator(max_concurrency=args.max_concurrency)
    report = await orchestrator.run_project(
        project_root=project_path,
        goal=goal,
        auto_approve=auto_approve,
    )
    json.dump(report.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return asyncio.run(_run_command(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
