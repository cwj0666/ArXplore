import importlib
from datetime import datetime
from typing import Any

CRONITER: Any = importlib.import_module("croniter").croniter


def previous_fire_time(expression: str, now: datetime) -> datetime:
    previous: datetime = CRONITER(expression, now).get_prev(datetime)
    return previous


def next_fire_time(expression: str, now: datetime) -> datetime:
    following: datetime = CRONITER(expression, now).get_next(datetime)
    return following
