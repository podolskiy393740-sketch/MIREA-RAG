from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.infrastructure.eval.pipeline import EvalResult


@dataclass(frozen=True)
class Distribution:
    count: int
    mean: float | None
    p10: float | None
    p50: float | None
    p90: float | None


@dataclass(frozen=True)
class EvalSummary:
    cases_total: int
    cases_scored: int
    success_rate_ge_4: float | None
    judge: Distribution
    rouge_1_mean: float | None
    rouge_l_mean: float | None
    latency_ms: Distribution
    human_fallback_rate: float


def summarize(results: Sequence[EvalResult]) -> EvalSummary:
    judge_scores = [float(r.judge_score) for r in results if r.judge_score is not None]
    success_rate_ge_4 = (
        sum(1 for s in judge_scores if s >= 4) / len(judge_scores) * 100.0 if judge_scores else None
    )
    fallback_count = sum(1 for r in results if r.needs_human_fallback)

    return EvalSummary(
        cases_total=len(results),
        cases_scored=len(judge_scores),
        success_rate_ge_4=success_rate_ge_4,
        judge=_distribution(judge_scores),
        rouge_1_mean=_mean(r.rouge_1 for r in results),
        rouge_l_mean=_mean(r.rouge_l for r in results),
        latency_ms=_distribution([r.latency_ms for r in results]),
        human_fallback_rate=(fallback_count / len(results) * 100.0) if results else 0.0,
    )


def format_summary(summary: EvalSummary) -> str:
    lines = [
        f"cases_total={summary.cases_total}",
        f"cases_scored={summary.cases_scored}",
        f"human_fallback_rate={summary.human_fallback_rate:.1f}%",
    ]
    if summary.success_rate_ge_4 is not None:
        lines.append(f"success_rate_ge_4={summary.success_rate_ge_4:.1f}%")
    if summary.judge.count:
        j = summary.judge
        lines.append(f"judge_score: mean={_fmt(j.mean)} p10={_fmt(j.p10)} p50={_fmt(j.p50)} p90={_fmt(j.p90)}")
    lines.append(f"rouge_1_mean={_fmt(summary.rouge_1_mean)}")
    lines.append(f"rouge_l_mean={_fmt(summary.rouge_l_mean)}")
    if summary.latency_ms.count:
        latency = summary.latency_ms
        lines.append(
            f"latency_ms: mean={_fmt(latency.mean)} p10={_fmt(latency.p10)} "
            f"p50={_fmt(latency.p50)} p90={_fmt(latency.p90)}"
        )
    return "\n".join(lines)


def _mean(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _distribution(values: Sequence[float]) -> Distribution:
    if not values:
        return Distribution(count=0, mean=None, p10=None, p50=None, p90=None)
    sorted_values = sorted(values)
    return Distribution(
        count=len(sorted_values),
        mean=sum(sorted_values) / len(sorted_values),
        p10=_percentile(sorted_values, 10),
        p50=_percentile(sorted_values, 50),
        p90=_percentile(sorted_values, 90),
    )


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _fmt(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.2f}"
