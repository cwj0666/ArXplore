from dataclasses import dataclass
from typing import Any, Self

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from arxplore.config import AppSettings
from arxplore.context import request_openai_api_key, request_openai_model

TEMPERATURE_FREE_MODEL_PREFIX = "gpt-5"
MISSING_API_KEY_MESSAGE = "OpenAI API 키가 없습니다."


class OpenAIKeyMissingError(RuntimeError):
    pass


def accepts_temperature(model: str) -> bool:
    return not model.startswith(TEMPERATURE_FREE_MODEL_PREFIX)


@dataclass(frozen=True, slots=True)
class ChatModelFactory:
    server_api_key: str | None
    default_model: str
    base_url: str | None = None

    @classmethod
    def from_settings(cls, settings: AppSettings, *, base_url: str | None = None) -> Self:
        return cls(
            server_api_key=settings.openai_api_key,
            default_model=settings.openai_model,
            base_url=base_url if base_url is not None else settings.openai_base_url,
        )

    def resolve_api_key(self) -> str | None:
        request_key = request_openai_api_key()
        if request_key:
            return request_key
        server_key = (self.server_api_key or "").strip()
        return server_key or None

    def is_available(self) -> bool:
        return self.resolve_api_key() is not None

    def resolve_model(self, model: str | None = None) -> str:
        return model or request_openai_model() or self.default_model

    def create(self, *, model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
        api_key = self.resolve_api_key()
        if api_key is None:
            raise OpenAIKeyMissingError(MISSING_API_KEY_MESSAGE)
        name = self.resolve_model(model)
        options: dict[str, Any] = {}
        if accepts_temperature(name):
            options["temperature"] = temperature
        return ChatOpenAI(
            model_name=name,
            openai_api_key=SecretStr(api_key),
            openai_api_base=self.base_url,
            stream_usage=True,
            **options,
        )
