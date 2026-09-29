from collections.abc import Callable
from datetime import date, datetime
from zoneinfo import ZoneInfo

type Today = Callable[[], date]


def today_in(timezone: ZoneInfo, *, now: datetime | None = None) -> date:
    return (now or datetime.now(timezone)).astimezone(timezone).date()


def zoned_today(timezone_name: str) -> Today:
    timezone = ZoneInfo(timezone_name)
    return lambda: today_in(timezone)
