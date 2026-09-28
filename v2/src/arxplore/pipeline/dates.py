from collections.abc import Callable
from datetime import date

type Today = Callable[[], date]


def container_today() -> date:
    return date.today()


def parse_optional_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    normalized = value.strip()
    return date.fromisoformat(normalized) if normalized else None
