from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.core.prompts.answer import ANSWER_QUESTION_PROMPT
from src.core.tracing import build_rag_answer_trace_config
from src.shared import get_runtime_openai_api_key, get_runtime_openai_model

from .tools import _format_context_papers, get_trending_papers_tool, search_paper_chunks_tool


def answer_question(
    question: str,
    *,
    context_papers: list[dict[str, Any]],
    chat_history: list | None = None,
    runtime: str = "dev",
    user: str | None = None,
) -> dict[str, Any]:
    model = get_runtime_openai_model()
    api_key = get_runtime_openai_api_key()
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.0,
    )
    chain = ANSWER_QUESTION_PROMPT | llm | StrOutputParser()
    formatted_context = _format_context_papers(context_papers)
    trace_config = build_rag_answer_trace_config(
        runtime=runtime,
        user=user,
    )
    answer_text = chain.invoke(
        {
            "question": question,
            "context_papers": formatted_context,
            "chat_history": chat_history or [],
        },
        config=trace_config,
    )
    return {
        "answer": answer_text,
        "source_papers": context_papers,
    }

def stream_answer_question(
    question: str,
    *,
    context_papers: list[dict[str, Any]],
    chat_history: list | None = None,
    runtime: str = "dev",
    user: str | None = None,
):
    model = get_runtime_openai_model()
    api_key = get_runtime_openai_api_key()
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.0,
    )
    chain = ANSWER_QUESTION_PROMPT | llm | StrOutputParser()
    formatted_context = _format_context_papers(context_papers)
    trace_config = build_rag_answer_trace_config(runtime=runtime, user=user)
    return chain.stream(
        {
            "question": question,
            "context_papers": formatted_context,
            "chat_history": chat_history or [],
        },
        config=trace_config,
    )

def agent_search(
    question: str, 
    *, 
    chat_history: list | None = None, 
    runtime: str = "dev", 
    user: str | None = None
) -> dict[str, Any]:
    model = get_runtime_openai_model()
    api_key = get_runtime_openai_api_key()
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.0,
    )
    tools = [search_paper_chunks_tool, get_trending_papers_tool]
    agent_executor = create_react_agent(llm, tools)
    messages = [
        ("system", "당신은 AI 논문 탐색 스마트 비서입니다. 사용자의 질문 의도를 파악하고, 적절한 도구를 최소 1번 이상 호출하여 가장 정확하고 친절한 답변을 생성하세요.\n"
                   "**[답변 형식 및 다듬기 규칙]**:\n"
                   "1. 어떠한 상황에서도 이모티콘(Emoji)을 절대 사용하지 마십시오.\n"
                   "2. 논문을 추천하거나 언급할 때는 반드시 아래의 마크다운 하이퍼링크 형식을 엄격하게 준수하십시오:\n"
                   "   - **[{논문 제목}]({URL})** — 이 논문의 핵심 내용이나 추천 사유 (1~2줄로 깔끔하게 요약)\n"
                   "3. 단순한 줄글 나열을 피하고, 가독성을 극대화하기 위해 개요(Bullet points) 형식을 적극 활용하십시오.\n"
                   "반드시 뉴스 브리핑 전문가처럼 '존댓말(~합니다, ~습니다)'로 정돈되게 출력하세요."),
    ]
    if chat_history:
        for role, content in chat_history:
            messages.append((role, content))
    messages.append(("user", question))
    trace_config = build_rag_answer_trace_config(runtime=runtime, user=user)
    result = agent_executor.invoke({"messages": messages}, config=trace_config)
    messages_out = result.get("messages", [])
    answer_text = messages_out[-1].content if messages_out else "답변을 가져올 수 없습니다."
    return {
        "answer": answer_text,
    }

def stream_agent_search(
    question: str, 
    *, 
    chat_history: list | None = None, 
    runtime: str = "dev", 
    user: str | None = None
):
    model = get_runtime_openai_model()
    api_key = get_runtime_openai_api_key()
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.0,
    )
    tools = [search_paper_chunks_tool, get_trending_papers_tool]
    agent_executor = create_react_agent(llm, tools)
    messages = [
        ("system", "당신은 AI 논문 탐색 스마트 비서입니다. 사용자의 질문 의도를 파악하고, 적절한 도구를 최소 1번 이상 호출하여 가장 정확하고 친절한 답변을 생성하세요.\n"
                   "**[답변 형식 및 다듬기 규칙]**:\n"
                   "1. 어떠한 상황에서도 이모티콘(Emoji)을 절대 사용하지 마십시오.\n"
                   "2. 논문을 추천하거나 언급할 때는 반드시 아래의 마크다운 하이퍼링크 형식을 엄격하게 준수하십시오:\n"
                   "   - **[{논문 제목}]({URL})** — 이 논문을 추천하는 사유 및 핵심 내용 (1~2줄 요약)\n"
                   "3. 단순한 줄글 나열을 피하고, 가독성을 극대화하기 위해 개요(Bullet points) 형식을 적극 활용하십시오.\n"
                   "반드시 뉴스 브리핑 전문가처럼 '존댓말(~합니다, ~습니다)'로 정돈되게 출력하세요."),
    ]
    if chat_history:
        for role, content in chat_history:
            messages.append((role, content))
    messages.append(("user", question))
    trace_config = build_rag_answer_trace_config(runtime=runtime, user=user)
    for chunk, _ in agent_executor.stream({"messages": messages}, config=trace_config, stream_mode="messages"):
        from langchain_core.messages import AIMessageChunk
        if isinstance(chunk, AIMessageChunk) and not chunk.tool_calls:
            if chunk.content:
                yield chunk.content
