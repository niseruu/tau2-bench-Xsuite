"""Utilities for the PELNI e-Billing domain."""

from tau2.utils.utils import DATA_DIR

PELNI_EBILL_DATA_DIR = DATA_DIR / "tau2" / "domains" / "pelni_ebill"
PELNI_EBILL_DB_PATH = PELNI_EBILL_DATA_DIR / "db.json"
PELNI_EBILL_POLICY_PATH = PELNI_EBILL_DATA_DIR / "policy.md"
PELNI_EBILL_TASK_SET_PATH = PELNI_EBILL_DATA_DIR / "tasks.json"
