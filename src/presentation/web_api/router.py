from __future__ import annotations

import json as _json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.application.use_cases import AnswerQuestionUseCase, NO_ANSWER_FOUND_TEXT
from src.domain.entities import Query, UserContext
from src.infrastructure.llm.prompt_templates import (
    INSUFFICIENT_DATA_MARKER,
    build_prompt,
)
from src.infrastructure.storage.postgres.fulltext_repository import PostgresFullTextRepository
from src.infrastructure.storage.postgres.session import get_session
from src.infrastructure.storage.postgres.vector_repository import PostgresVectorRepository
from src.presentation.web_api.schemas import AskRequest, AskResponse

api_router = APIRouter()


def _build_user_ctx(body: AskRequest) -> UserContext | None:
    if body.user_context and (body.user_context.course or body.user_context.faculty):
        return UserContext(course=body.user_context.course, faculty=body.user_context.faculty)
    return None


@api_router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    embedder = request.app.state.embedder
    llm = request.app.state.llm

    async with get_session() as fts_session, get_session() as vector_session:
        use_case = AnswerQuestionUseCase(
            fulltext_store=PostgresFullTextRepository(fts_session),
            vector_store=PostgresVectorRepository(vector_session),
            embedder=embedder,
            llm=llm,
        )
        answer = await use_case.execute(Query(text=body.question, user_context=_build_user_ctx(body)))

    return AskResponse(answer=answer.text, needs_human_fallback=answer.needs_human_fallback)


@api_router.post("/ask/stream")
async def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    """SSE-стриминг: токены идут клиенту сразу, не ждя полного ответа.
    Формат событий: {token: str} | {replace: str} | {done: true, needs_human_fallback: bool}"""
    embedder = request.app.state.embedder
    llm = request.app.state.llm
    query = Query(text=body.question, user_context=_build_user_ctx(body))

    async def event_stream():
        def sse(data: dict) -> str:
            return f"data: {_json.dumps(data, ensure_ascii=False)}\n\n"

        # Retrieval (сессии закрываются сразу после — они нужны только здесь)
        async with get_session() as fts_session, get_session() as vector_session:
            use_case = AnswerQuestionUseCase(
                fulltext_store=PostgresFullTextRepository(fts_session),
                vector_store=PostgresVectorRepository(vector_session),
                embedder=embedder,
                llm=llm,
            )
            context_chunks = await use_case._retrieve(query)

        if not context_chunks or llm is None:
            yield sse({"token": NO_ANSWER_FOUND_TEXT})
            yield sse({"done": True, "needs_human_fallback": True})
            return

        prompt = build_prompt(query.text, context_chunks, query.user_context)
        full_text = ""
        try:
            if hasattr(llm, "stream"):
                async for token in llm.stream(prompt):
                    full_text += token
                    yield sse({"token": token})
            else:
                # LLM без стриминга — отдаём одним куском
                full_text = await llm.generate(prompt)
                yield sse({"token": full_text})
        except Exception:
            yield sse({"token": NO_ANSWER_FOUND_TEXT})
            yield sse({"done": True, "needs_human_fallback": True})
            return

        is_fallback = INSUFFICIENT_DATA_MARKER in full_text
        if is_fallback:
            yield sse({"replace": NO_ANSWER_FOUND_TEXT})
        yield sse({"done": True, "needs_human_fallback": is_fallback})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
