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


def _normalize_url_prefix(value: str) -> str:
    prefix = value.strip().strip("/")
    if not prefix:
        raise ImproperlyConfigured("DJANGO_ADMIN_PATH must not be empty.")
    return f"{prefix}/"


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
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "papers",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
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
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
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

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

LANGUAGE_CODE = "ko-kr"

TIME_ZONE = "Asia/Seoul"

USE_I18N = True

USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [("frontend", FRONTEND_DIST_DIR)] if FRONTEND_DIST_DIR.exists() else []

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

SECURE_COOKIES = _env_bool("DJANGO_SECURE_COOKIES", False)
SESSION_COOKIE_SECURE = SECURE_COOKIES
CSRF_COOKIE_SECURE = SECURE_COOKIES
if SECURE_COOKIES:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"

ADMIN_ENABLED = _env_bool("DJANGO_ADMIN_ENABLED", False)
ADMIN_PATH = _normalize_url_prefix(os.getenv("DJANGO_ADMIN_PATH", "admin/"))

DEMO_MODE = _env_bool("DEMO_MODE", True)

SESSION_KEY_ENCRYPTION_KEY = os.getenv("SESSION_KEY_ENCRYPTION_KEY", "")

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
RATE_LIMIT_AUTH_PER_MINUTE = _env_int("RATE_LIMIT_AUTH_PER_MINUTE", 10)
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
