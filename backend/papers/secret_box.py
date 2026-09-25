"""세션에 저장하는 사용자 API 키의 대칭 암호화 (Fernet).

암호화 키는 SESSION_KEY_ENCRYPTION_KEY(없으면 SECRET_KEY)에서 HKDF-SHA256으로 32바이트를 유도한다.
`v2.` 접두사가 없는 값(이전 평문 저장분 등)은 복호화하지 않고 무효로 본다.
"""

from __future__ import annotations

import base64

from cryptography.fernet import Fernet
from cryptography.fernet import InvalidToken as FernetInvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from django.conf import settings

TOKEN_PREFIX = "v2."
_HKDF_SALT = b"arxplore.session-api-key"
_HKDF_INFO = b"fernet"


class InvalidToken(ValueError):
    pass


def encrypt_secret(plaintext: str) -> str:
    return TOKEN_PREFIX + _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str) -> str:
    if not isinstance(token, str) or not token.startswith(TOKEN_PREFIX):
        raise InvalidToken("unsupported token format")
    try:
        return _fernet().decrypt(token[len(TOKEN_PREFIX):].encode("ascii")).decode("utf-8")
    except (FernetInvalidToken, UnicodeError) as exc:
        raise InvalidToken("invalid token") from exc


def _fernet() -> Fernet:
    material = (getattr(settings, "SESSION_KEY_ENCRYPTION_KEY", "") or settings.SECRET_KEY).encode("utf-8")
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=_HKDF_SALT, info=_HKDF_INFO).derive(material)
    return Fernet(base64.urlsafe_b64encode(key))
