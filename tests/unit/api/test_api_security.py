import pytest
from starlette.requests import Request

from arxplore.api.security import (
    CsrfFailure,
    CsrfRejectedError,
    TrustedOrigins,
    check_csrf,
    client_ip,
    host_domain,
    is_allowed_host,
    is_valid_csrf_token,
    new_csrf_token,
)

TOKEN = "a" * 32
TRUSTED = TrustedOrigins.parse(["http://localhost:5173", "https://*.example.com"])


def request(
    method: str = "POST",
    *,
    headers: dict[str, str] | None = None,
    scheme: str = "http",
    client: tuple[str, int] | None = ("172.18.0.5", 5000),
) -> Request:
    raw_headers = [(b"host", b"localhost")] + [
        (name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": method,
        "scheme": scheme,
        "path": "/papers/x/analyze/",
        "query_string": b"",
        "headers": raw_headers,
        "client": client,
        "server": ("localhost", 80),
    }
    return Request(scope)


def failure(req: Request, *, trust_forwarded_proto: bool = False) -> CsrfFailure | None:
    try:
        check_csrf(req, trusted_origins=TRUSTED, trust_forwarded_proto=trust_forwarded_proto)
    except CsrfRejectedError as exc:
        return exc.failure
    return None


def test_new_tokens_are_32_alphanumeric_characters() -> None:
    token = new_csrf_token()

    assert is_valid_csrf_token(token)
    assert token != new_csrf_token()
    assert not is_valid_csrf_token("a" * 31)
    assert not is_valid_csrf_token("-" * 32)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, CsrfFailure.NO_COOKIE),
        ({"cookie": "csrftoken=short"}, CsrfFailure.BAD_COOKIE),
        ({"cookie": f"csrftoken={TOKEN}"}, CsrfFailure.TOKEN_MISSING),
        ({"cookie": f"csrftoken={TOKEN}", "x-csrftoken": ""}, CsrfFailure.TOKEN_INCORRECT),
        ({"cookie": f"csrftoken={TOKEN}", "x-csrftoken": "b" * 32}, CsrfFailure.TOKEN_INCORRECT),
        ({"cookie": f"csrftoken={TOKEN}", "x-csrftoken": TOKEN}, None),
        (
            {"cookie": f"csrftoken={TOKEN}", "x-csrftoken": TOKEN, "origin": "http://evil.example"},
            CsrfFailure.BAD_ORIGIN,
        ),
        ({"origin": "http://evil.example"}, CsrfFailure.BAD_ORIGIN),
        ({"cookie": f"csrftoken={TOKEN}", "x-csrftoken": TOKEN, "origin": "http://localhost:5173"}, None),
        ({"cookie": f"csrftoken={TOKEN}", "x-csrftoken": TOKEN, "origin": "http://localhost"}, None),
        ({"cookie": f"csrftoken={TOKEN}", "x-csrftoken": TOKEN, "origin": "https://api.example.com"}, None),
        ({"cookie": f"csrftoken={TOKEN}", "x-csrftoken": TOKEN, "origin": "https://example.com"}, None),
        (
            {"cookie": f"csrftoken={TOKEN}", "x-csrftoken": TOKEN, "origin": "http://api.example.com"},
            CsrfFailure.BAD_ORIGIN,
        ),
    ],
)
def test_csrf_rules(headers: dict[str, str], expected: CsrfFailure | None) -> None:
    assert failure(request(headers=headers)) is expected


def test_safe_methods_skip_the_check() -> None:
    assert failure(request("GET")) is None


def test_same_origin_follows_the_forwarded_scheme_only_when_trusted() -> None:
    headers = {"cookie": f"csrftoken={TOKEN}", "x-csrftoken": TOKEN, "origin": "https://localhost"}

    assert failure(request(headers=headers)) is CsrfFailure.BAD_ORIGIN
    forwarded = {**headers, "x-forwarded-proto": "https"}
    assert failure(request(headers=forwarded)) is CsrfFailure.BAD_ORIGIN
    assert failure(request(headers=forwarded), trust_forwarded_proto=True) is None


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", "localhost"),
        ("LOCALHOST:8001", "localhost"),
        ("example.com.", "example.com"),
        ("[::1]:80", "[::1]"),
        ("bad host", ""),
        ("", ""),
    ],
)
def test_host_domain(host: str, expected: str) -> None:
    assert host_domain(host) == expected


@pytest.mark.parametrize(
    ("host", "allowed", "expected"),
    [
        ("localhost:8001", ["localhost", "127.0.0.1"], True),
        ("evil.example", ["localhost", "127.0.0.1"], False),
        ("anything.test", ["*"], True),
        ("api.example.com", [".example.com"], True),
        ("example.com", [".example.com"], True),
        ("badexample.com", [".example.com"], False),
        ("", ["*"], False),
    ],
)
def test_allowed_hosts_follow_the_django_rules(host: str, allowed: list[str], expected: bool) -> None:
    assert is_allowed_host(host, allowed) is expected


def test_client_ip_uses_the_first_value_of_the_configured_header() -> None:
    headers = {"x-real-ip": "203.0.113.9", "x-forwarded-for": "203.0.113.8, 10.0.0.1"}

    assert client_ip(request(headers=headers), "X-Real-IP") == "203.0.113.9"
    assert client_ip(request(headers=headers), "X-Forwarded-For") == "203.0.113.8"


def test_client_ip_falls_back_to_the_socket_address_then_unknown() -> None:
    assert client_ip(request(headers={"x-real-ip": "203.0.113.9"}), "none") == "172.18.0.5"
    assert client_ip(request(headers={"x-real-ip": "203.0.113.9"}), "NONE") == "172.18.0.5"
    assert client_ip(request(headers={"x-real-ip": " "}), "X-Real-IP") == "172.18.0.5"
    assert client_ip(request(), "X-Real-IP") == "172.18.0.5"
    assert client_ip(request(client=None), "X-Real-IP") == "unknown"
