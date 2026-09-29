import re
import secrets
import string
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

CSRF_COOKIE_NAME = "csrftoken"
CSRF_HEADER_NAME = "x-csrftoken"
CSRF_COOKIE_MAX_AGE = 60 * 60 * 24 * 7 * 52
CSRF_COOKIE_SAMESITE: Literal["lax"] = "lax"
CSRF_TOKEN_LENGTH = 32
CSRF_ALLOWED_CHARS = string.ascii_letters + string.digits
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
FORWARDED_PROTO_HEADER = "x-forwarded-proto"
SECURE_SCHEME = "https"
DISABLED_IP_HEADER = "none"
UNKNOWN_CLIENT_IP = "unknown"
WILDCARD_HOST = "*"
HOST_PATTERN = re.compile(r"([a-z0-9.-]+|\[[a-f0-9]*:[a-f0-9.:]+\])(?::([0-9]+))?")
SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"same-origin"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"x-frame-options", b"DENY"),
)


class CsrfFailure(StrEnum):
    BAD_ORIGIN = "bad_origin"
    NO_COOKIE = "no_cookie"
    BAD_COOKIE = "bad_cookie"
    TOKEN_MISSING = "token_missing"
    TOKEN_INCORRECT = "token_incorrect"


class CsrfRejectedError(Exception):
    def __init__(self, failure: CsrfFailure) -> None:
        super().__init__(failure.value)
        self.failure = failure


def new_csrf_token() -> str:
    return "".join(secrets.choice(CSRF_ALLOWED_CHARS) for _ in range(CSRF_TOKEN_LENGTH))


def is_valid_csrf_token(token: str | None) -> bool:
    return (
        token is not None
        and len(token) == CSRF_TOKEN_LENGTH
        and all(character in CSRF_ALLOWED_CHARS for character in token)
    )


def set_csrf_cookie(response: Response, token: str, *, secure: bool) -> None:
    response.set_cookie(
        CSRF_COOKIE_NAME,
        token,
        max_age=CSRF_COOKIE_MAX_AGE,
        expires=CSRF_COOKIE_MAX_AGE,
        path="/",
        secure=secure,
        httponly=False,
        samesite=CSRF_COOKIE_SAMESITE,
    )


def csrf_cookie_token(request: Request) -> str:
    token = request.cookies.get(CSRF_COOKIE_NAME)
    return token if token is not None and is_valid_csrf_token(token) else new_csrf_token()


def request_scheme(request: Request, *, trust_forwarded_proto: bool) -> str:
    if trust_forwarded_proto:
        forwarded = request.headers.get(FORWARDED_PROTO_HEADER, "")
        return SECURE_SCHEME if forwarded.split(",", 1)[0].strip().lower() == SECURE_SCHEME else "http"
    return request.url.scheme


def is_same_domain(host: str, pattern: str) -> bool:
    if not pattern:
        return False
    pattern = pattern.lower()
    return (pattern[0] == "." and (host.endswith(pattern) or host == pattern[1:])) or pattern == host


@dataclass(frozen=True, slots=True)
class TrustedOrigins:
    exact: frozenset[str]
    subdomains: dict[str, tuple[str, ...]]

    @classmethod
    def parse(cls, origins: Iterable[str]) -> TrustedOrigins:
        exact: set[str] = set()
        subdomains: dict[str, list[str]] = {}
        for origin in origins:
            if "*" not in origin:
                exact.add(origin)
                continue
            parts = urlsplit(origin)
            subdomains.setdefault(parts.scheme, []).append(parts.netloc.lstrip("*"))
        return cls(frozenset(exact), {scheme: tuple(hosts) for scheme, hosts in subdomains.items()})

    def allows(self, origin: str, *, request_origin: str) -> bool:
        if origin == request_origin or origin in self.exact:
            return True
        try:
            parts = urlsplit(origin)
        except ValueError:
            return False
        return any(is_same_domain(parts.netloc, host) for host in self.subdomains.get(parts.scheme, ()))


def check_csrf(request: Request, *, trusted_origins: TrustedOrigins, trust_forwarded_proto: bool) -> None:
    if request.method in SAFE_METHODS:
        return
    origin = request.headers.get("origin")
    if origin is not None:
        own_origin = f"{request_scheme(request, trust_forwarded_proto=trust_forwarded_proto)}://{request_host(request)}"
        if not trusted_origins.allows(origin, request_origin=own_origin):
            raise CsrfRejectedError(CsrfFailure.BAD_ORIGIN)
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if cookie_token is None:
        raise CsrfRejectedError(CsrfFailure.NO_COOKIE)
    if not is_valid_csrf_token(cookie_token):
        raise CsrfRejectedError(CsrfFailure.BAD_COOKIE)
    header_token = request.headers.get(CSRF_HEADER_NAME)
    if header_token is None:
        raise CsrfRejectedError(CsrfFailure.TOKEN_MISSING)
    if not secrets.compare_digest(header_token.encode(), cookie_token.encode()):
        raise CsrfRejectedError(CsrfFailure.TOKEN_INCORRECT)


def request_host(request: Request) -> str:
    return request.headers.get("host", "")


def host_domain(host: str) -> str:
    match = HOST_PATTERN.fullmatch(host.lower())
    if match is None:
        return ""
    return match.group(1).removesuffix(".")


def is_allowed_host(host: str, allowed_hosts: Sequence[str]) -> bool:
    domain = host_domain(host)
    if not domain:
        return False
    return any(pattern == WILDCARD_HOST or is_same_domain(domain, pattern) for pattern in allowed_hosts)


def client_ip(request: Request, header_name: str) -> str:
    name = header_name.strip()
    if name and name.lower() != DISABLED_IP_HEADER:
        forwarded = (request.headers.get(name) or "").split(",")[0].strip()
        if forwarded:
            return forwarded
    if request.client is not None and request.client.host:
        return request.client.host
    return UNKNOWN_CLIENT_IP


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                present = {name.lower() for name, _ in message.get("headers", [])}
                extra = [(name, value) for name, value in SECURITY_HEADERS if name not in present]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class TrustedHostMiddleware:
    def __init__(self, app: ASGIApp, *, allowed_hosts: Sequence[str], rejection: Callable[[], Response]) -> None:
        self.app = app
        self.allowed_hosts = tuple(allowed_hosts)
        self.rejection = rejection

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or is_allowed_host(Headers(scope=scope).get("host", ""), self.allowed_hosts):
            await self.app(scope, receive, send)
            return
        await self.rejection()(scope, receive, send)
