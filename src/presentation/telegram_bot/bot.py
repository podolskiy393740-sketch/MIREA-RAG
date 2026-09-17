import asyncio

from aiogram import Bot, Dispatcher

from src.application.use_cases import AnswerQuestionUseCase
from src.presentation.telegram_bot.config import BotSettings
from src.presentation.telegram_bot.handlers import router


async def main() -> None:
    settings = BotSettings()
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    # Ports не подключены — use case работает в режиме human-fallback,
    # пока не готова инфраструктура (Postgres/эмбеддинги/LLM).
    answer_question = AnswerQuestionUseCase()

    await dispatcher.start_polling(bot, answer_question=answer_question)


if __name__ == "__main__":
    asyncio.run(main())
