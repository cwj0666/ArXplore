import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"

for path in (BACKEND_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_TEST_ENV_DEFAULTS = {
    "DJANGO_SETTINGS_MODULE": "arxplore_web.test_settings",
    "DJANGO_SECRET_KEY": "test-only-secret-key",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_DB": "arxplore_test",
    "POSTGRES_USER": "arxplore_test",
    "POSTGRES_PASSWORD": "arxplore_test",
    "PROD_POSTGRES_HOST": "localhost",
    "MONGO_HOST": "localhost",
    "OPENAI_API_KEY": "sk-test-dummy",
    "LANGSMITH_TRACING": "false",
    "LANGSMITH_API_KEY": "",
}
for key, value in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires a real PostgreSQL at TEST_DATABASE_URL; skipped when it is unset",
    )


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not url.startswith(("postgresql://", "postgres://")):
        pytest.skip("TEST_DATABASE_URL (postgresql://...) is not set")
    return url
