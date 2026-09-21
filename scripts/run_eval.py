"""Прогоняет эталонный набор question,answer через реальный
AnswerQuestionUseCase и считает ROUGE + LLM-judge (методология —
CLAUDE.md, референс подхода — eval-пайплайн chernenko-s/mirea-rag).

По умолчанию — held-out набор Славы (data/external/mirea_rag_slava_test.csv,
49 кейсов), отдельный от того, что уже проиндексировано в базе.

Использование (при поднятом Postgres и заполненной .env):
    python -m scripts.run_eval
    python -m scripts.run_eval --csv path/to/other.csv --limit 10 --concurrency 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from src.application.use_cases import AnswerQuestionUseCase
from src.domain.entities import Answer, Query
from src.infrastructure.embeddings.qwen_local_embedder import QwenLocalEmbedder
from src.infrastructure.eval.dataset import load_eval_cases_from_csv
from src.infrastructure.eval.llm_judge import LLMJudge
from src.infrastructure.eval.pipeline import evaluate_dataset
from src.infrastructure.eval.report import format_summary, summarize
from src.infrastructure.llm.openai_llm import OpenAIClient
from src.infrastructure.storage.postgres.fulltext_repository import PostgresFullTextRepository
from src.infrastructure.storage.postgres.session import get_session
from src.infrastructure.storage.postgres.vector_repository import PostgresVectorRepository

_DEFAULT_CSV = "data/eval/curated_15.csv"
_SLAVA_CSV = "data/external/mirea_rag_slava_test.csv"


async def run(csv_path: str, limit: int | None, concurrency: int, output_path: str | None) -> None:
    cases = load_eval_cases_from_csv(csv_path)
    if limit is not None:
        cases = cases[:limit]
    print(f"Кейсов в наборе: {len(cases)}")

    embedder = QwenLocalEmbedder()
    llm = OpenAIClient()
    judge = LLMJudge(OpenAIClient())  # тот же дефолт модели, что и генерация

    async def answer_fn(question: str) -> Answer:
        # Отдельная сессия на каждый вызов (и внутри — на каждый порт):
        # _retrieve() в AnswerQuestionUseCase гоняет FTS/вектор параллельно,
        # одна AsyncSession конкурентность не переживает (см. project memory).
        async with get_session() as fts_session, get_session() as vector_session:
            use_case = AnswerQuestionUseCase(
                fulltext_store=PostgresFullTextRepository(fts_session),
                vector_store=PostgresVectorRepository(vector_session),
                embedder=embedder,
                llm=llm,
            )
            return await use_case.execute(Query(text=question))

    def on_result(completed: int, total: int, result) -> None:
        status = "ERROR" if result.error else ("fallback" if result.needs_human_fallback else "ok")
        judge_part = f"judge={result.judge_score}" if result.judge_score is not None else "judge=?"
        f_part = f"F={result.faithfulness:.2f}" if result.faithfulness is not None else "F=?"
        ar_part = f"AR={result.answer_relevance:.2f}" if result.answer_relevance is not None else "AR=?"
        cr_part = f"CR={result.context_recall:.2f}" if result.context_recall is not None else "CR=?"
        print(f"[{completed}/{total}] {status} {judge_part} {f_part} {ar_part} {cr_part} :: {result.question[:55]}")

    results = await evaluate_dataset(cases, answer_fn, judge, concurrency=concurrency, on_result=on_result)

    print()
    print(format_summary(summarize(results)))

    if output_path is not None:
        # Полные результаты по каждому кейсу (не только сводка) — для
        # детального разбора ошибок (например, какие именно 8 вопросов
        # ушли в фолбек) без повторного дорогого прогона.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"\nПолные результаты сохранены в {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=_DEFAULT_CSV, help="Путь к CSV (question,answer)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число кейсов")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Параллельных кейсов (платный OpenAI выдержит больше, чем free-тир OpenRouter, но лимиты аккаунта неизвестны — по умолчанию осторожно)",
    )
    parser.add_argument("--output", default=None, help="Путь для сохранения полных результатов в JSON")
    args = parser.parse_args()

    asyncio.run(run(args.csv, args.limit, args.concurrency, args.output))
