"""Run state written to disk: checkpoints, JSON records and code revision."""

import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    # Write then rename, so a disconnect never leaves a truncated checkpoint
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def write_json(path: Path, content: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2))
    temporary.replace(path)


def git_revision() -> str | None:
    # A "-dirty" suffix records that uncommitted changes were part of the run
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
