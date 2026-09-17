# Сравнение вариантов эмбеддингов

Открытый вопрос из `CLAUDE.md` — решение не принято, этот документ только
сравнивает варианты для команды/куратора. Собрано веб-поиском, сентябрь
2026.

## Важный нюанс

Три варианта из `CLAUDE.md` не полностью независимы — **Qwen3-Embedding
доступен и через OpenRouter API, и как self-hosted веса**. Это не третья
отдельная опция, а модель, которую можно попробовать в обоих режимах.

## Сравнение

| | **OpenRouter API** | **Self-hosted (sentence-transformers)** | **Qwen3-Embedding** |
|---|---|---|---|
| Что это | Единый API поверх чужих моделей (OpenAI text-embedding-3, voyage-4, Qwen3-Embedding и др.) | Локальный inference: multilingual-e5, BAAI/bge-m3, deepvk/USER-bge-m3 (рус.-тюн), rubert-tiny2/turbo | Открытые веса (0.6B/4B/8B) — можно и через OpenRouter, и локально |
| Задержка | Сетевой round-trip к внешнему API | Только локальный inference — ниже, если есть CPU/GPU под рукой | Зависит от пути (API или локально) |
| Стоимость | Копейки за токен (Qwen3-Embedding-8B на OpenRouter ≈ $0.01/M ток., 4B ≈ $0.02/M — дороже за меньшую модель, не опечатка, так ценит провайдер) | Бесплатно, но нужны вычислительные ресурсы (хостинг) | И то, и другое |
| Приватность | Контент уходит на сторонний сервер | Данные не покидают инфраструктуру | Зависит от пути |
| Качество на русском | Сильное (voyage-4, Qwen3-Embedding-8B — топ ruMTEB/MTEB-multilingual) | От слабого (rubert-tiny2, но очень быстрый) до сильного (BGE-M3/USER-bge-m3, тоже топ ruMTEB) | 8B — №1 в MTEB-multilingual (июнь 2025) |
| Инфраструктура | Никакой — только API-ключ | Нужен процесс инференса (минимум CPU, для крупных моделей — GPU) | Зависит от пути |
| Размерность | 768/1536/2048... — обычно настраиваемая | Фиксированная под модель (384–1024) | Гибкая (Matryoshka): до 1024/2560/4096, можно обрезать |

## Важные нюансы для проекта

1. **Приватность здесь ниже риска, чем в вопросе персонализации.** Эмбеддим
   публичные документы (положения, регламенты МИРЭА), а не личные данные
   студентов — отправка на OpenRouter не пересекается с privacy-блокером
   из `CLAUDE.md` про личный кабинет. Это разные вопросы, не стоит их
   смешивать перед куратором.
2. **Требование куратора "снижение задержки"** — сильный аргумент за
   self-hosted: нет сетевого round-trip к внешнему API ни на индексации,
   ни на каждый запрос пользователя.
3. **`EMBEDDING_DIM=768`** в текущей схеме (`.env`, Alembic-миграция) —
   заглушка. Какую бы модель ни выбрали, нужно явно сверить её размерность
   (rubert-tiny2 — 312, multilingual-e5-base — 768, BGE-M3 — 1024,
   Qwen3-Embedding — гибко) и поправить `.env`/миграцию под неё.
4. **rubert-tiny2/turbo** — неплохой демо-аргумент для акселератора ("наш
   прототип работает на CPU без внешних API и без GPU"), но качество
   retrieval заметно ниже, чем у BGE-M3/Qwen — риск похожих, но не точных
   чанков.

## Рекомендация (не решение — за командой и куратором)

Для курсовой версии (первокурсники, ограниченный набор документов) —
**self-hosted `deepvk/USER-bge-m3`** (BGE-M3, дообученная на русском):
хорошее качество на ruMTEB, детерминированная задержка без внешнего API,
бесплатно, работает на CPU (медленнее GPU, но для прототипа с ограниченным
трафиком — приемлемо).

Если на защите нужно явно показать сравнение self-hosted vs. API-подхода —
Qwen3-Embedding позволяет продемонстрировать оба пути на одной модели
переключением конфига (`EmbedderPort` уже спроектирован как заменяемый
порт, см. `ARCHITECTURE.md`) — может быть хорошим слайдом.

## Источники

- [Qwen3 Embedding 8B — API Pricing & Providers | OpenRouter](https://openrouter.ai/qwen/qwen3-embedding-8b)
- [Text Embedding Models | OpenRouter](https://openrouter.ai/collections/embedding-models)
- [Qwen3 Embedding 4B — API Pricing & Providers | OpenRouter](https://openrouter.ai/qwen/qwen3-embedding-4b)
- [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://qwenlm.github.io/blog/qwen3-embedding/)
- [GitHub - avidale/encodechka](https://github.com/avidale/encodechka)
- [ruMTEB benchmark and Russian embedding model design](https://aclanthology.org/2025.naacl-long.12.pdf)
- [sergeyzh/rubert-tiny-lite · Hugging Face](https://huggingface.co/sergeyzh/rubert-tiny-lite)
