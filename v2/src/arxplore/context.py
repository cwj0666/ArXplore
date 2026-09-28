from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from arxplore.domain.retrieval import RetrievalResult


@dataclass(frozen=True, slots=True)
class OpenAIRuntime:
    api_key: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    query: str
    arxiv_id: str | None
    limit: int
    result: RetrievalResult


NO_OPENAI_OVERRIDE = OpenAIRuntime()

_openai_runtime: ContextVar[OpenAIRuntime] = ContextVar("arxplore_openai_runtime", default=NO_OPENAI_OVERRIDE)
_retrieval_observers: ContextVar[tuple[list[RetrievalObservation], ...]] = ContextVar(
    "arxplore_retrieval_observers", default=()
)


@contextmanager
def override_openai_runtime(*, api_key: str | None = None, model: str | None = None) -> Iterator[None]:
    current = _openai_runtime.get()
    token = _openai_runtime.set(
        OpenAIRuntime(
            api_key=current.api_key if api_key is None else api_key,
            model=current.model if model is None else model,
        )
    )
    try:
        yield
    finally:
        _openai_runtime.reset(token)


def openai_runtime() -> OpenAIRuntime:
    return _openai_runtime.get()


def request_openai_api_key() -> str | None:
    normalized = (_openai_runtime.get().api_key or "").strip()
    return normalized or None


def request_openai_model() -> str | None:
    return _openai_runtime.get().model or None


@contextmanager
def observe_retrieval() -> Iterator[list[RetrievalObservation]]:
    observations: list[RetrievalObservation] = []
    token = _retrieval_observers.set((*_retrieval_observers.get(), observations))
    try:
        yield observations
    finally:
        _retrieval_observers.reset(token)


def record_retrieval(observation: RetrievalObservation) -> None:
    for observations in _retrieval_observers.get():
        observations.append(observation)
