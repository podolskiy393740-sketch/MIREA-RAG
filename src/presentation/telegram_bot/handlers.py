from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from src.application.use_cases import AnswerQuestionUseCase
from src.domain.entities import Query
from src.domain.ports import EmbedderPort, LLMPort
from src.infrastructure.storage.postgres.fulltext_repository import PostgresFullTextRepository
from src.infrastructure.storage.postgres.session import get_session
from src.infrastructure.storage.postgres.vector_repository import PostgresVectorRepository

router = Router()


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(
        "Привет! Я RAG-помощник МИРЭА (тестовая версия). "
        "Задайте вопрос — постараюсь ответить на основе документов вуза."
    )


@router.message()
async def handle_question(message: Message, embedder: EmbedderPort, llm: LLMPort | None) -> None:
    if not message.text:
        return

    # Свежая сессия/use case на каждое сообщение: AsyncSession не рассчитан
    # на конкурентное использование из разных апдейтов бота, а embedder/llm
    # (тяжёлая модель, HTTP-клиент) — одни и те же на всё время жизни бота,
    # передаются через workflow_data (см. bot.py), а не создаются заново.
    async with get_session() as session:
        answer_question = AnswerQuestionUseCase(
            fulltext_store=PostgresFullTextRepository(session),
            vector_store=PostgresVectorRepository(session),
            embedder=embedder,
            llm=llm,
        )
        answer = await answer_question.execute(Query(text=message.text))

    await message.answer(answer.text)
