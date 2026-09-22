from __future__ import annotations

from fastapi import APIRouter, Request

from src.application.use_cases import AnswerQuestionUseCase
from src.domain.entities import Query, UserContext
from src.infrastructure.storage.postgres.fulltext_repository import PostgresFullTextRepository
from src.infrastructure.storage.postgres.session import get_session
from src.infrastructure.storage.postgres.vector_repository import PostgresVectorRepository
from src.presentation.web_api.schemas import AskRequest, AskResponse

api_router = APIRouter()


@api_router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    embedder = request.app.state.embedder
    llm = request.app.state.llm

    user_ctx: UserContext | None = None
    if body.user_context and (body.user_context.course or body.user_context.faculty):
        user_ctx = UserContext(
            course=body.user_context.course,
            faculty=body.user_context.faculty,
        )

    async with get_session() as fts_session, get_session() as vector_session:
        use_case = AnswerQuestionUseCase(
            fulltext_store=PostgresFullTextRepository(fts_session),
            vector_store=PostgresVectorRepository(vector_session),
            embedder=embedder,
            llm=llm,
        )
        answer = await use_case.execute(Query(text=body.question, user_context=user_ctx))

    return AskResponse(answer=answer.text, needs_human_fallback=answer.needs_human_fallback)
