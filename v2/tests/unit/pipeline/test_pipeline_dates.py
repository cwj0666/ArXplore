from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from arxplore.pipeline.dates import today_in, zoned_today

SEOUL = ZoneInfo("Asia/Seoul")


@pytest.mark.parametrize(
    "utc_now,expected",
    [
        (datetime(2026, 9, 29, 14, 59, tzinfo=UTC), date(2026, 9, 29)),
        (datetime(2026, 9, 29, 15, 0, tzinfo=UTC), date(2026, 9, 30)),
        (datetime(2026, 9, 29, 23, 30, tzinfo=UTC), date(2026, 9, 30)),
    ],
)
def test_today_is_the_seoul_date_whatever_the_host_clock_zone(utc_now: datetime, expected: date) -> None:
    assert today_in(SEOUL, now=utc_now) == expected


def test_zoned_today_reads_the_clock_each_time() -> None:
    today = zoned_today("Asia/Seoul")
    assert today() == datetime.now(SEOUL).date()
    assert zoned_today("UTC")() == datetime.now(UTC).date()
