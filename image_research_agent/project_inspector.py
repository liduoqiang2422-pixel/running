from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Iterable

from .models import ProjectSnapshot

EXCLUDED_DIR_NAMES = {".git", ".pytest_cache", "__pycache__", ".venv", "venv"}
DEPENDENCY_FILE_NAMES = {
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "env.yaml",
    "pyproject.toml",
}
TRAIN_SUFFIXES = {".py", ".sh", ".ps1"}
TEST_SUFFIXES = {".py", ".sh", ".ps1"}
CONFIG_SUFFIXES = {".yml", ".yaml"}
CHECKPOINT_PATTERN = re.compile(r"(?i)\b(checkpoint|ckpt|pretrain|resume|weights?)\b")
DATASET_PATTERN = re.compile(
    r"(?i)\b(dataset|data_dir|dataroot|train_dataset|val_dataset|test_dataset)\b"
)


def inspect_project(project_root: Path) -> ProjectSnapshot:
    resolved_root = project_root.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"Project path does not exist or is not a directory: {project_root}")

    files = tuple(_iter_files(resolved_root))
    readme_files = _relative_paths(
        resolved_root,
        [file_path for file_path in files if file_path.name.lower().startswith("readme")],
    )
    dependency_files = _relative_paths(
        resolved_root,
        [file_path for file_path in files if file_path.name.lower() in DEPENDENCY_FILE_NAMES],
    )
    train_entrypoints = _relative_paths(
        resolved_root,
        [
            file_path
            for file_path in files
            if file_path.suffix.lower() in TRAIN_SUFFIXES
            and "train" in file_path.stem.lower()
        ],
    )
    test_entrypoints = _relative_paths(
        resolved_root,
        [
            file_path
            for file_path in files
            if file_path.suffix.lower() in TEST_SUFFIXES
            and any(
                marker in file_path.stem.lower()
                for marker in ("test", "eval", "infer")
            )
        ],
    )
    config_files = _relative_paths(
        resolved_root,
        [file_path for file_path in files if file_path.suffix.lower() in CONFIG_SUFFIXES],
    )
    inspected_text_files = (*readme_files, *config_files)
    checkpoint_references = _collect_matching_lines(
        resolved_root,
        inspected_text_files,
        CHECKPOINT_PATTERN,
        limit=6,
    )
    dataset_hints = _collect_matching_lines(
        resolved_root,
        inspected_text_files,
        DATASET_PATTERN,
        limit=6,
    )
    tags = infer_project_tags(resolved_root.name, _read_joined_text(resolved_root, inspected_text_files))
    return ProjectSnapshot(
        project_name=resolved_root.name,
        project_root=str(resolved_root),
        readme_files=readme_files,
        dependency_files=dependency_files,
        train_entrypoints=train_entrypoints,
        test_entrypoints=test_entrypoints,
        config_files=config_files,
        checkpoint_references=checkpoint_references,
        dataset_hints=dataset_hints,
        tags=tags,
    )


def build_suggested_commands(snapshot: ProjectSnapshot) -> dict[str, str | None]:
    environment_command = _environment_command(snapshot)
    train_config = _first_named(snapshot.config_files, ("*train*.yml", "*train*.yaml"))
    test_config = _first_named(snapshot.config_files, ("*test*.yml", "*test*.yaml"))
    return {
        "environment": environment_command,
        "train": _script_command(_first_or_none(snapshot.train_entrypoints), train_config),
        "test": _script_command(_first_or_none(snapshot.test_entrypoints), test_config),
    }


def infer_metric_targets(snapshot: ProjectSnapshot) -> tuple[str, ...]:
    metrics: list[str] = []
    if "restoration" in snapshot.tags:
        metrics.extend(["PSNR", "SSIM"])
    if "watermark" in snapshot.tags:
        metrics.extend(["bit_accuracy", "tamper_localization"])
    if not metrics:
        metrics.append("task_success_rate")
    metrics.extend(["config_coverage", "checkpoint_resolution"])

    unique_metrics: list[str] = []
    for metric in metrics:
        if metric not in unique_metrics:
            unique_metrics.append(metric)
    return tuple(unique_metrics)


def infer_project_tags(project_name: str, text_blob: str) -> tuple[str, ...]:
    tags: list[str] = []
    lowered_blob = f"{project_name}\n{text_blob}".lower()
    if any(keyword in lowered_blob for keyword in ("restore", "restoration", "denoise", "derain", "dehaze")):
        tags.append("restoration")
    if any(keyword in lowered_blob for keyword in ("watermark", "tamper", "bit accuracy", "localization")):
        tags.append("watermark")
    if "lightning" in lowered_blob:
        tags.append("pytorch-lightning")
    if not tags:
        tags.append("generic-research")
    return tuple(tags)


def _iter_files(project_root: Path) -> Iterable[Path]:
    for file_path in project_root.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in file_path.parts):
            continue
        yield file_path


def _relative_paths(project_root: Path, files: Iterable[Path]) -> tuple[str, ...]:
    return tuple(
        sorted(file_path.relative_to(project_root).as_posix() for file_path in files)
    )


def _collect_matching_lines(
    project_root: Path,
    relative_paths: Iterable[str],
    pattern: re.Pattern[str],
    limit: int,
) -> tuple[str, ...]:
    collected: list[str] = []
    for relative_path in relative_paths:
        file_path = project_root / relative_path
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            if not pattern.search(line):
                continue
            normalized_line = line.strip()
            if not normalized_line:
                continue
            collected.append(f"{relative_path}: {normalized_line}")
            if len(collected) >= limit:
                return tuple(collected)
    return tuple(collected)


def _read_joined_text(project_root: Path, relative_paths: Iterable[str]) -> str:
    chunks: list[str] = []
    for relative_path in relative_paths:
        file_path = project_root / relative_path
        try:
            chunks.append(file_path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def _environment_command(snapshot: ProjectSnapshot) -> str | None:
    dependency_files = set(snapshot.dependency_files)
    for candidate in ("environment.yml", "environment.yaml", "env.yaml"):
        if candidate in dependency_files:
            return f"conda env create -f {candidate}"
    if "requirements.txt" in dependency_files:
        return "python -m pip install -r requirements.txt"
    if "pyproject.toml" in dependency_files:
        return "python -m pip install -e ."
    return None


def _script_command(entrypoint: str | None, config_path: str | None) -> str | None:
    if entrypoint is None:
        return None
    suffix = Path(entrypoint).suffix.lower()
    if suffix == ".py":
        command = f"python {entrypoint}"
    elif suffix == ".sh":
        command = f"bash {entrypoint}"
    elif suffix == ".ps1":
        command = f"powershell -File {entrypoint}"
    else:
        return entrypoint
    if config_path and suffix == ".py":
        return f"{command} -opt {config_path}"
    return command


def _first_named(values: Iterable[str], patterns: tuple[str, ...]) -> str | None:
    for value in values:
        if any(fnmatch.fnmatch(value.lower(), pattern.lower()) for pattern in patterns):
            return value
    return None


def _first_or_none(values: Iterable[str]) -> str | None:
    for value in values:
        return value
    return None
