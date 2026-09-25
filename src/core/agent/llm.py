from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.shared import get_runtime_openai_api_key, get_runtime_openai_model


def build_chat_llm(*, temperature: float = 0.0) -> ChatOpenAI:
    """요청 범위의 키·모델로 ChatOpenAI를 만든다. 키 문자열을 그대로 넘기므로 요청 사이에 클라이언트를 공유하지 않는다."""
    api_key = get_runtime_openai_api_key()
    if not api_key:
        raise ValueError("OpenAI API 키가 없습니다.")
    model = get_runtime_openai_model()
    kwargs: dict = {"model": model, "api_key": api_key}
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)
