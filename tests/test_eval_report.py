from src.infrastructure.eval.pipeline import EvalResult
from src.infrastructure.eval.report import summarize


def _result(judge_score, rouge_1=0.5, rouge_l=0.5, latency_ms=100.0, needs_human_fallback=False) -> EvalResult:
    return EvalResult(
        question="q",
        ideal_answer="a",
        model_answer="m",
        rouge_1=rouge_1,
        rouge_l=rouge_l,
        judge_score=judge_score,
        judge_reason=None,
        latency_ms=latency_ms,
        needs_human_fallback=needs_human_fallback,
    )


def test_success_rate_counts_scores_ge_4():
    results = [_result(5), _result(4), _result(3), _result(2), _result(1)]

    summary = summarize(results)

    assert summary.cases_scored == 5
    assert summary.success_rate_ge_4 == 40.0  # 2 из 5


def test_unscored_cases_excluded_from_judge_distribution():
    results = [_result(5), _result(None)]

    summary = summarize(results)

    assert summary.cases_total == 2
    assert summary.cases_scored == 1
    assert summary.judge.mean == 5.0


def test_human_fallback_rate():
    results = [_result(5, needs_human_fallback=True), _result(5, needs_human_fallback=False)]

    summary = summarize(results)

    assert summary.human_fallback_rate == 50.0


def test_rouge_means_computed_correctly():
    results = [_result(5, rouge_1=1.0, rouge_l=0.5), _result(5, rouge_1=0.0, rouge_l=0.5)]

    summary = summarize(results)

    assert summary.rouge_1_mean == 0.5
    assert summary.rouge_l_mean == 0.5


def test_empty_results_do_not_crash():
    summary = summarize([])

    assert summary.cases_total == 0
    assert summary.human_fallback_rate == 0.0
    assert summary.judge.count == 0
