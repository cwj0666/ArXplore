import importlib
from collections.abc import Callable, Sequence

from arxplore.retrieval.service import QueryEmbedder

TOKEN_ENCODING_MODEL = "text-embedding-3-large"
ESTIMATED_CHARS_PER_TOKEN = 4
TIKTOKEN_METHOD = "tiktoken cl100k_base"
ESTIMATED_METHOD = "추정(글자 수 / 4)"


class MemoEmbedder:
    def __init__(self, inner: QueryEmbedder) -> None:
        self.inner = inner
        self.cache: dict[tuple[str, ...], list[list[float]]] = {}

    def is_available(self) -> bool:
        return self.inner.is_available()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        key = tuple(texts)
        if key not in self.cache:
            self.cache[key] = await self.inner.embed(list(texts))
        return self.cache[key]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // ESTIMATED_CHARS_PER_TOKEN)


def token_counter() -> tuple[Callable[[str], int], str]:
    try:
        tiktoken = importlib.import_module("tiktoken")
        encoding = tiktoken.encoding_for_model(TOKEN_ENCODING_MODEL)
    except Exception:
        return _estimate_tokens, ESTIMATED_METHOD

    def count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    return count, TIKTOKEN_METHOD


class CountingEmbedder:
    def __init__(self, inner: QueryEmbedder, counter: Callable[[str], int] | None = None) -> None:
        self.inner = inner
        if counter is None:
            self._count, self.token_method = token_counter()
        else:
            self._count, self.token_method = counter, "주입"
        self.calls = 0
        self.texts = 0
        self.tokens = 0

    def is_available(self) -> bool:
        return self.inner.is_available()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        self.tokens += sum(self._count(text) for text in texts)
        return await self.inner.embed(texts)
