from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from src.shared import AppSettings, get_settings
from src.shared.settings import get_runtime_openai_api_key_override


class EmbeddingClient:
    """문자열 목록을 OpenAI 임베딩 벡터로 변환한다.

    키는 호출 시점에 고른다. 요청 범위 키(`override_openai_runtime`, 웹 요청의 사용자 세션 키)가
    있으면 그 키를, 없으면 서버 `OPENAI_API_KEY`를 쓴다.
    """

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        client: OpenAI | None = None,
        model_name: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.openai_embedding_model
        self.dimensions = dimensions or self.settings.openai_embedding_dimensions
        self.batch_size = max(1, batch_size or self.settings.embedding_batch_size)
        self._client = client
        self._cached_client: tuple[str, OpenAI] | None = None

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """텍스트 목록을 벡터 목록으로 변환한다."""
        normalized_texts = [self._sanitize_text(str(text or "").strip()) for text in texts]
        if not normalized_texts:
            return []

        client = self._get_client()
        embeddings: list[list[float]] = []

        for start in range(0, len(normalized_texts), self.batch_size):
            batch = normalized_texts[start : start + self.batch_size]
            request_kwargs = {
                "model": self.model_name,
                "input": batch,
            }
            if self.model_name.startswith("text-embedding-3"):
                request_kwargs["dimensions"] = self.dimensions

            response = client.embeddings.create(**request_kwargs)
            embeddings.extend([list(item.embedding) for item in response.data])

        return embeddings

    @staticmethod
    def _sanitize_text(value: str) -> str:
        return "".join(char for char in value if not 0xD800 <= ord(char) <= 0xDFFF)

    def resolve_api_key(self) -> str | None:
        """요청 범위 키 > 서버 키 순서로 임베딩에 쓸 키를 고른다. 둘 다 없으면 None."""
        override = get_runtime_openai_api_key_override()
        if override:
            return override
        server_key = str(self.settings.openai_api_key or "").strip()
        return server_key or None

    def is_available(self) -> bool:
        return self._client is not None or self.resolve_api_key() is not None

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        api_key = self.resolve_api_key()
        if not api_key:
            raise ValueError("임베딩에 쓸 OpenAI API 키가 없습니다(사용자 키, OPENAI_API_KEY 모두 비어 있음).")
        if self._cached_client is None or self._cached_client[0] != api_key:
            self._cached_client = (api_key, OpenAI(api_key=api_key))
        return self._cached_client[1]
