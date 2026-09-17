from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from src.application.use_cases import AnswerQuestionUseCase
from src.domain.entities import Query

router = Router()


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(
        "Привет! Я RAG-помощник МИРЭА (тестовая версия). "
        "Задайте вопрос — постараюсь ответить на основе документов вуза."
    )


@router.message()
async def handle_question(message: Message, answer_question: AnswerQuestionUseCase) -> None:
    if not message.text:
        return

    answer = await answer_question.execute(Query(text=message.text))
    await message.answer(answer.text)
