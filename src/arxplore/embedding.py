from collections.abc import Sequence
from typing import Self

from openai import AsyncOpenAI

from arxplore.config import AppSettings
from arxplore.context import request_openai_api_key
from arxplore.text import strip_surrogates

DEFAULT_EMBEDDING_BATCH_SIZE = 64
DIMENSIONS_MODEL_PREFIX = "text-embedding-3"


class EmbeddingKeyMissingError(ValueError):
    pass


def clean_embedding_input(text: str) -> str:
    return strip_surrogates(text.strip())


class EmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = max(1, batch_size)
        self.base_url = base_url
        self._server_api_key = api_key
        self._clients: dict[str, AsyncOpenAI] = {}

    @classmethod
    def from_settings(cls, settings: AppSettings, *, base_url: str | None = None) -> Self:
        return cls(
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            api_key=settings.openai_api_key,
            base_url=base_url if base_url is not None else settings.openai_base_url,
        )

    def resolve_api_key(self) -> str | None:
        request_key = request_openai_api_key()
        if request_key:
            return request_key
        server_key = (self._server_api_key or "").strip()
        return server_key or None

    def is_available(self) -> bool:
        return self.resolve_api_key() is not None

    def sends_dimensions(self) -> bool:
        return self.model.startswith(DIMENSIONS_MODEL_PREFIX)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = [clean_embedding_input(text) for text in texts]
        if not inputs:
            return []
        client = self._client()
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start : start + self.batch_size]
            if self.sends_dimensions():
                response = await client.embeddings.create(model=self.model, input=batch, dimensions=self.dimensions)
            else:
                response = await client.embeddings.create(model=self.model, input=batch)
            vectors.extend(list(item.embedding) for item in response.data)
        return vectors

    async def aclose(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.close()

    def _client(self) -> AsyncOpenAI:
        api_key = self.resolve_api_key()
        if api_key is None:
            raise EmbeddingKeyMissingError(
                "임베딩에 쓸 OpenAI API 키가 없습니다(사용자 키, OPENAI_API_KEY 모두 비어 있음)."
            )
        client = self._clients.get(api_key)
        if client is None:
            client = self._clients[api_key] = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        return client
