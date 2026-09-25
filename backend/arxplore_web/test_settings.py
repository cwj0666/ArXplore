"""Settings for tests that must run without PostgreSQL or a real .env."""

import os

_TEST_ENV_DEFAULTS = {
    "DJANGO_SECRET_KEY": "test-only-secret-key",
    "DJANGO_DEBUG": "False",
    "DJANGO_ALLOWED_HOSTS": "testserver,localhost,127.0.0.1",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_DB": "arxplore_test",
    "POSTGRES_USER": "arxplore_test",
    "POSTGRES_PASSWORD": "arxplore_test",
}
for _key, _value in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

ALLOW_PLACEHOLDER_SECRET_KEY = True

from .settings import *  # noqa: E402,F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

STATICFILES_DIRS = []

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "arxplore-tests",
    }
}
RATE_LIMIT_ENABLED = False
