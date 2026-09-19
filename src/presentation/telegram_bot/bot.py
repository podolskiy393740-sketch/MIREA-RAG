import asyncio
import logging

from aiogram import Bot, Dispatcher

from src.infrastructure.embeddings.qwen_local_embedder import QwenLocalEmbedder
from src.infrastructure.llm.factory import build_llm
from src.presentation.telegram_bot.config import BotSettings
from src.presentation.telegram_bot.handlers import router


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = BotSettings()
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    # Embedder и llm — по одному на всё время жизни бота: модель
    # эмбеддингов грузится один раз, а LLM-клиент просто держит HTTP-сессию.
    # AnswerQuestionUseCase с БД-сессией строится заново на каждое
    # сообщение (см. handlers.py) — AsyncSession не рассчитан на
    # конкурентное использование из разных апдейтов.
    embedder = QwenLocalEmbedder()
    llm = build_llm()  # режим local/prototype — см. llm/factory.py (ФЗ-152)

    await dispatcher.start_polling(bot, embedder=embedder, llm=llm)


if __name__ == "__main__":
    asyncio.run(main())
