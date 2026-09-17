import asyncio

from aiogram import Bot, Dispatcher
from pydantic import ValidationError

from src.domain.ports import LLMPort
from src.infrastructure.embeddings.qwen_local_embedder import QwenLocalEmbedder
from src.infrastructure.llm.openrouter_llm import OpenRouterLLMClient, OpenRouterSettings
from src.presentation.telegram_bot.config import BotSettings
from src.presentation.telegram_bot.handlers import router


def _try_build_llm() -> LLMPort | None:
    try:
        settings = OpenRouterSettings()
    except ValidationError:
        return None

    if not settings.openrouter_api_key:
        # Пустая строка — валидный str для pydantic, ValidationError тут
        # не сработает. Это ожидаемо на машинах без настроенного ключа —
        # бот продолжает работать в режиме human-фолбека, а не падает.
        return None

    return OpenRouterLLMClient(api_key=settings.openrouter_api_key, model=settings.openrouter_model)


async def main() -> None:
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
    llm = _try_build_llm()

    await dispatcher.start_polling(bot, embedder=embedder, llm=llm)


if __name__ == "__main__":
    asyncio.run(main())
