import asyncio

from aiogram import Bot, Dispatcher
from pydantic import ValidationError

from src.domain.ports import LLMPort
from src.infrastructure.embeddings.qwen_local_embedder import QwenLocalEmbedder
from src.infrastructure.llm.openai_llm import OpenAIClient, OpenAISettings
from src.infrastructure.llm.openrouter_llm import OpenRouterLLMClient, OpenRouterSettings
from src.presentation.telegram_bot.config import BotSettings
from src.presentation.telegram_bot.handlers import router


def _try_build_llm() -> LLMPort | None:
    """OpenAI — основной путь генерации (решение команды, платный ключ).
    OpenRouter — рабочая альтернатива (например, для сравнения на защите),
    подключается только если OpenAI не настроен. Без обоих ключей бот
    продолжает работать в режиме human-фолбека, а не падает при старте."""
    try:
        openai_settings = OpenAISettings()
        if openai_settings.openai_api_key:  # пустая строка — валидный str для pydantic
            return OpenAIClient(api_key=openai_settings.openai_api_key, model=openai_settings.openai_model)
    except ValidationError:
        pass

    try:
        openrouter_settings = OpenRouterSettings()
        if openrouter_settings.openrouter_api_key:
            return OpenRouterLLMClient(
                api_key=openrouter_settings.openrouter_api_key, model=openrouter_settings.openrouter_model
            )
    except ValidationError:
        pass

    return None


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
