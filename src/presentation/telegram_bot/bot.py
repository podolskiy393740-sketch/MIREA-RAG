import asyncio

from aiogram import Bot, Dispatcher

from src.infrastructure.embeddings.qwen_local_embedder import QwenLocalEmbedder
from src.presentation.telegram_bot.config import BotSettings
from src.presentation.telegram_bot.handlers import router


async def main() -> None:
    settings = BotSettings()
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    # Один embedder на всё время жизни бота — модель (Qwen3-Embedding)
    # грузится один раз при первом вызове, а не на каждое сообщение.
    # LLM ещё не выбрана (открытый вопрос, см. CLAUDE.md), поэтому
    # AnswerQuestionUseCase (строится на каждое сообщение — см.
    # handlers.py) пока всегда отвечает human-фолбеком с найденными
    # источниками вместо сгенерированного текста.
    embedder = QwenLocalEmbedder()

    await dispatcher.start_polling(bot, embedder=embedder)


if __name__ == "__main__":
    asyncio.run(main())
