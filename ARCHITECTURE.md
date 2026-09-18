# Архитектура — RAG-помощник МИРЭА

Контекст и зафиксированные решения — см. `CLAUDE.md`. Здесь — как это
раскладывается на код.

## Принцип

Чистая архитектура (ориентир: chernenko-s/mirea-rag), четыре слоя,
зависимости направлены внутрь (presentation → application → domain,
infrastructure реализует порты domain/application):

```
src/
  domain/            # сущности и порты (интерфейсы), без внешних зависимостей
  application/       # use cases — оркестрация сценариев
  infrastructure/    # конкретные реализации портов (БД, LLM, эмбеддеры)
  presentation/       # адаптеры входа: Telegram-бот, веб (FastAPI)
```

Смысл разделения именно для этого проекта: **два открытых вопроса
(эмбеддинги, персонализация) не должны требовать переписывания retrieval**,
когда решение будет принято. Поэтому они с самого начала — порты
(интерфейсы), а не прямые вызовы конкретной библиотеки/API.

## domain/

Сущности (dataclasses/pydantic, без логики ввода-вывода):

- `Document` — исходный документ (источник, тип: html-страница/PDF-регламент/…)
- `Chunk` — фрагмент документа + метаданные (источник, позиция, тип чанкинга)
- `Query` — вопрос пользователя (+ опционально `UserContext`, см. ниже)
- `RetrievedChunk` — чанк + скор + откуда (vector/FTS/RRF-ранг)
- `Answer` — ответ + список источников (для UX «на основе каких документов»)
  + флаг уверенности (для фолбека на человека)

Порты (`Protocol`/ABC), реализации — в `infrastructure/`:

- `ChunkerPort` — `chunk(document: Document) -> list[Chunk]`
- `EmbedderPort` — `embed(texts: list[str]) -> list[Vector]`
- `VectorStorePort` — upsert/search по эмбеддингам (pgvector)
- `FullTextStorePort` — upsert/search по FTS (Postgres tsvector)
- `LLMPort` — генерация ответа по промпту + контексту
- `UserContextPort` *(заглушка под персонализацию, см. ниже)*

## application/ (use cases)

- `IngestDocumentUseCase` — принимает `Document`, выбирает стратегию
  чанкинга через `ChunkerSelector` (см. ниже), эмбеддит, пишет в
  `VectorStorePort` и `FullTextStorePort`.
- `AnswerQuestionUseCase` — основной сценарий:
  1. `FullTextStorePort.search()` + `VectorStorePort.search()` параллельно
  2. `rrf_fuse(fts_results, vector_results)` → объединённый ранжированный
     список (см. ниже про RRF)
  3. сборка контекста для LLM с учётом лимита контекстного окна
     (обрезка/приоритизация чанков по RRF-рангу)
  4. `LLMPort.generate(prompt, context)` → ответ + self-check уверенности
  5. если уверенность ниже порога → `Answer` с флагом `needs_human_fallback`
     вместо галлюцинации
- `EvaluateAnswerUseCase` — прогон эталонного набора вопросов через
  ROUGE + LLM-judge (ориентир — подход из chernenko-s/mirea-rag).

Presentation-слои (Telegram, веб) вызывают **одни и те же use cases** —
это даёт переносимость на веб-виджет для акселератора без дублирования
логики.

## infrastructure/

**Chunking (гибрид, выбор по типу документа):**
```
infrastructure/chunking/
  structure_aware_chunker.py   # по заголовкам/абзацам — для размеченных страниц
  recursive_chunker.py         # recursive character splitting — для сплошного текста/PDF
  chunker_selector.py          # ChunkerPort: маршрутизация по Document.doc_type
```
`ChunkerSelector` реализует `ChunkerPort` и внутри делегирует одному из
двух — это и есть «не или/или, а гибрид» из CLAUDE.md.

**Embeddings — решение принято: Qwen3-Embedding-0.6B, self-hosted через
sentence-transformers** (см. `docs/embeddings-comparison.md` — сравнение
вариантов; без внешнего API, без сетевой задержки на каждый чанк/запрос,
соответствует требованию куратора по задержке).
```
infrastructure/embeddings/
  qwen_local_embedder.py   # EmbedderPort, sentence-transformers, EMBEDDING_DIM=1024
```
Реализация — за портом `EmbedderPort` (см. `domain/ports.py`), поэтому
если решение по модели пересмотрят — меняется одна реализация, retrieval-
код не трогается.

**Storage (Postgres + pgvector):**
```
infrastructure/storage/postgres/
  models.py            # SQLAlchemy-модели: documents, chunks (vector + tsvector колонка)
  vector_repository.py     # VectorStorePort
  fulltext_repository.py   # FullTextStorePort
  migrations/           # Alembic
```

**Retrieval fusion:**
```
infrastructure/retrieval/rrf.py
```
RRF: `score(d) = Σ 1 / (k + rank_i(d))` по спискам FTS и vector, k — константа
сглаживания (обычно 60). Не требует нормализации разных шкал скоров —
поэтому устойчив к разнородности FTS-скора и косинусной близости. Даёт
устойчивость к опечаткам в запросах, потому что документ может быть высоко
в FTS-списке даже при слабом векторном сходстве, и наоборот.

**LLM / генерация — решение принято: GPT напрямую через OpenAI API**
(`gpt-5.4-nano`, платный ключ команды; сравнение вариантов см. в истории
решений проекта — self-hosted для чата почти всегда требует GPU, иначе
задержка неприемлема, в отличие от self-hosted эмбеддингов):
```
infrastructure/llm/
  prompt_templates.py           # шаблон с системным промптом + грамотной укладкой источников
  openai_compatible_client.py   # общая retry-логика для OpenAI-совместимых chat/completions API
  openai_llm.py                 # LLMPort через OpenAI (основной путь)
  openrouter_llm.py             # LLMPort через OpenRouter (рабочая альтернатива, не подключена по умолчанию)
```
Требования куратора реализуются здесь как конкретные механизмы, не декларации:
- *снижение галлюцинаций* — промпт требует опоры только на переданный
  контекст + явную фразу «недостаточно данных», если ответа нет в чанках;
  плюс post-hoc проверка, что цитируемые факты действительно есть в
  источниках (можно эвристикой на этапе курсовой, LLM-judge — на этапе eval)
- *работа с контекстным окном* — приоритизация чанков по RRF-рангу при
  формировании промпта, обрезка по токен-бюджету, а не по числу чанков
- *снижение задержки* — параллельные FTS/vector запросы (asyncio.gather),
  кэш эмбеддинга запроса, стриминг ответа в Telegram

**Eval:**
```
infrastructure/eval/
  rouge_evaluator.py
  llm_judge_evaluator.py
```

## presentation/

```
presentation/telegram_bot/   # aiogram-хендлеры → AnswerQuestionUseCase
presentation/web_api/        # FastAPI-роуты → те же use cases (для акселератора)
```
Старт — только Telegram (внутреннее тестирование). Веб-слой добавляется
позже как ещё один адаптер поверх тех же use cases, без изменения domain/
application.

## Персонализация — задел, не реализация

Пока не решено (privacy-гипотеза не готова, нужна отдельная UX-ветка для
неавторизованных пользователей), поэтому в архитектуре **только заготовка**:

- `Query.user_context: UserContext | None = None` — опциональное поле с
  самого начала, чтобы `AnswerQuestionUseCase` не пришлось переписывать
  под появление персонализации.
- `UserContextPort` — порт, сейчас без реализации (или заглушка,
  возвращающая `None`).
- Авторизация (если будет — полноценный вход через личный кабинет, не
  виджет) живёт целиком в `presentation/web_api/`, domain/application про
  неё не знают ничего, кроме опционального `UserContext`.

Это единственное, что делается по персонализации до отдельного решения
по privacy-гипотезе — заготовка интерфейса, не реализация доступа к
личному кабинету.

## Стек и зависимости (справочно)

Python 3.11+, FastAPI, SQLAlchemy + Alembic, asyncpg, pgvector,
aiogram (Telegram), pydantic. Точный список фиксируется в `pyproject.toml`
по мере реализации — не входит в этот документ, чтобы не расходиться с
кодом.
