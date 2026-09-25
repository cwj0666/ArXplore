from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from src.shared import AppSettings, get_settings


class EmbeddingClient:
    """문자열 목록을 OpenAI 임베딩 벡터로 변환한다."""

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

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client
