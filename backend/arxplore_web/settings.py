"""Django settings for arxplore_web project."""

import os
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

from arxplore_web.bootstrap import configure_environment

configure_environment()

from src.shared import build_django_postgres_database_config, get_settings

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
APP_SETTINGS = get_settings()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ImproperlyConfigured(f"{name} must be an integer.") from exc


DEBUG = os.getenv("DJANGO_DEBUG", "").strip().lower() == "true"
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG=False.")
    SECRET_KEY = "django-insecure-dev-only-arxplore-secret-key"
_ALLOW_PLACEHOLDER_SECRET_KEY = bool(
    getattr(sys.modules.get("arxplore_web.test_settings"), "ALLOW_PLACEHOLDER_SECRET_KEY", False)
)
if SECRET_KEY.startswith("change-me") and not _ALLOW_PLACEHOLDER_SECRET_KEY:
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY is still the .env.example placeholder (starts with 'change-me'). "
        'Set a real secret key, e.g. python -c "import secrets; print(secrets.token_urlsafe(50))".'
    )
FRONTEND_PORT = os.getenv("FRONTEND_PORT", "5173")

ALLOWED_HOSTS = _env_csv("DJANGO_ALLOWED_HOSTS", ["*"] if DEBUG else ["localhost", "127.0.0.1"])
CSRF_TRUSTED_ORIGINS = _env_csv(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
    [
        "http://localhost",
        "http://127.0.0.1",
        f"http://localhost:{FRONTEND_PORT}",
        f"http://127.0.0.1:{FRONTEND_PORT}",
    ],
)

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "papers",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "arxplore_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "arxplore_web.wsgi.application"

try:
    DATABASES = {
        "default": build_django_postgres_database_config(APP_SETTINGS),
    }
except ValueError as exc:
    raise ImproperlyConfigured(str(exc)) from exc

LANGUAGE_CODE = "ko-kr"

TIME_ZONE = "Asia/Seoul"

USE_I18N = True

USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [("frontend", FRONTEND_DIST_DIR)] if FRONTEND_DIST_DIR.exists() else []

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

SECURE_COOKIES = _env_bool("DJANGO_SECURE_COOKIES", False)
CSRF_COOKIE_SECURE = SECURE_COOKIES
if SECURE_COOKIES:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
CSRF_COOKIE_SAMESITE = "Lax"
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"

REDIS_URL = os.getenv("REDIS_URL", "").strip()
if REDIS_URL:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "arxplore-default",
        }
    }

RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_LLM_PER_MINUTE = _env_int("RATE_LIMIT_LLM_PER_MINUTE", 30)
RATE_LIMIT_DETAIL_PER_MINUTE = _env_int("RATE_LIMIT_DETAIL_PER_MINUTE", 60)
RATE_LIMIT_IP_HEADER = os.getenv("RATE_LIMIT_IP_HEADER", "X-Real-IP").strip()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {
        "papers": {"handlers": ["console"], "level": "INFO"},
    },
}
