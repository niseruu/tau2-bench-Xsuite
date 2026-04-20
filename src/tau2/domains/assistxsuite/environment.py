"""Environment for the AssistXSuite domain."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.assistxsuite.data_model import AssistXSuiteDB
from tau2.domains.assistxsuite.tools import AssistXSuiteTools
from tau2.domains.assistxsuite.utils import (
    ASSISTXSUITE_DB_PATH,
    ASSISTXSUITE_POLICY_PATH,
    ASSISTXSUITE_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_db() -> AssistXSuiteDB:
    """Load the AssistXSuite database."""

    return AssistXSuiteDB.load(ASSISTXSUITE_DB_PATH)


def get_environment(
    db: Optional[AssistXSuiteDB] = None,
    solo_mode: bool = False,
) -> Environment:
    """Build the AssistXSuite environment."""

    if solo_mode:
        raise ValueError("assistxsuite domain does not support solo mode")
    if db is None:
        db = get_db()
    tools = AssistXSuiteTools(db)
    with open(ASSISTXSUITE_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="assistxsuite",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Load tasks for the AssistXSuite domain."""

    tasks = load_file(ASSISTXSUITE_TASK_SET_PATH)
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
    """Load task splits for the AssistXSuite domain."""

    split_file = (
        Path(ASSISTXSUITE_TASK_SET_PATH).parent
        / f"split_{Path(ASSISTXSUITE_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)

