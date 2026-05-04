"""Environment for the PELNI e-Billing domain."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.pelni_ebill.data_model import PelniEbillDB
from tau2.domains.pelni_ebill.tools import PelniEbillTools
from tau2.domains.pelni_ebill.utils import (
    PELNI_EBILL_DB_PATH,
    PELNI_EBILL_POLICY_PATH,
    PELNI_EBILL_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_db() -> PelniEbillDB:
    """Load the PELNI e-Billing database."""

    return PelniEbillDB.load(PELNI_EBILL_DB_PATH)


def get_environment(
    db: Optional[PelniEbillDB] = None,
    solo_mode: bool = False,
) -> Environment:
    """Build the PELNI e-Billing environment."""

    if solo_mode:
        raise ValueError("pelni_ebill domain does not support solo mode")
    if db is None:
        db = get_db()
    tools = PelniEbillTools(db)
    with open(PELNI_EBILL_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="pelni_ebill",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Load tasks for the PELNI e-Billing domain."""

    tasks = load_file(PELNI_EBILL_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    """Load task splits for the PELNI e-Billing domain."""

    split_file = (
        Path(PELNI_EBILL_TASK_SET_PATH).parent
        / f"split_{Path(PELNI_EBILL_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
