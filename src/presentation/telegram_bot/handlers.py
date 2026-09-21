from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from src.application.use_cases import AnswerQuestionUseCase
from src.domain.entities import Query
from src.domain.ports import EmbedderPort, LLMPort
from src.infrastructure.storage.postgres.fulltext_repository import PostgresFullTextRepository
from src.infrastructure.storage.postgres.session import get_session
from src.infrastructure.storage.postgres.vector_repository import PostgresVectorRepository

router = Router()

_EXAMPLE_QUESTIONS = [
    "Как перевестись на другую специальность?",
    "Какие документы нужны для академического отпуска?",
    "Как оформить социальную стипендию?",
    "Что делать при задолженности по предмету?",
]

_START_TEXT = """👋 Привет! Я RAG-помощник МИРЭА.

Задавай вопросы о поступлении, учёбе, документах и правилах вуза — отвечаю на основе официальных документов МИРЭА.

<b>Что умею:</b>
• Объяснять правила и регламенты вуза
• Помогать с вопросами про учёбу и документооборот
• Искать нужную информацию в официальных источниках

<b>Что не умею:</b>
• Получать данные из твоего личного кабинета
• Отвечать за актуальность информации (уточняй в деканате)

Просто напиши свой вопрос ⬇️"""


def _examples_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=q)] for q in _EXAMPLE_QUESTIONS],
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Введите вопрос...",
    )


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(
        _START_TEXT,
        parse_mode="HTML",
        reply_markup=_examples_keyboard(),
    )


@router.message()
async def handle_question(message: Message, embedder: EmbedderPort, llm: LLMPort | None) -> None:
    if not message.text:
        return

    # Отдельная сессия на каждый репозиторий, а не одна общая: retrieve()
    # внутри AnswerQuestionUseCase запускает FTS- и векторный поиск
    # параллельно (asyncio.gather) — одна AsyncSession не рассчитана на
    # конкурентное использование и падает с IllegalStateChangeError, если
    # её отдать в оба репозитория сразу. Плюс свежая сессия на каждое
    # сообщение бота: AsyncSession не рассчитан и на конкурентность между
    # разными апдейтами. embedder/llm — одни и те же на всё время жизни
    # бота, передаются через workflow_data (см. bot.py).
    async with get_session() as fts_session, get_session() as vector_session:
        answer_question = AnswerQuestionUseCase(
            fulltext_store=PostgresFullTextRepository(fts_session),
            vector_store=PostgresVectorRepository(vector_session),
            embedder=embedder,
            llm=llm,
        )
        answer = await answer_question.execute(Query(text=message.text))

    await message.answer(answer.text, reply_markup=ReplyKeyboardRemove())
