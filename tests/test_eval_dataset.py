from pathlib import Path

from src.infrastructure.eval.dataset import load_eval_cases_from_csv

_FIXTURE = Path(__file__).parent / "fixtures" / "sample_eval_dataset.csv"


def test_loads_all_rows():
    cases = load_eval_cases_from_csv(_FIXTURE)

    assert len(cases) == 2
    assert cases[0].question == "Когда стипендия?"
    assert cases[0].ideal_answer == "25 числа каждого месяца."
