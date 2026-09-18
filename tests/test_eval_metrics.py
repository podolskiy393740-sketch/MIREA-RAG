from src.infrastructure.eval.metrics import rouge_1_f1, rouge_l_f1


def test_identical_text_scores_one():
    assert rouge_1_f1("Стипендия выплачивается 25 числа", "Стипендия выплачивается 25 числа") == 1.0
    assert rouge_l_f1("Стипендия выплачивается 25 числа", "Стипендия выплачивается 25 числа") == 1.0


def test_completely_different_text_scores_zero():
    assert rouge_1_f1("Стипендия выплачивается 25 числа", "Общежитие находится на Вернадского") == 0.0


def test_both_empty_scores_one():
    assert rouge_1_f1("", "") == 1.0
    assert rouge_l_f1("", "") == 1.0


def test_one_empty_scores_zero():
    assert rouge_1_f1("текст", "") == 0.0
    assert rouge_1_f1("", "текст") == 0.0


def test_partial_overlap_is_between_zero_and_one():
    score = rouge_1_f1("Стипендия выплачивается 25 числа каждого месяца", "Стипендия приходит 25 числа")
    assert 0.0 < score < 1.0


def test_rouge_l_penalizes_word_order_more_than_rouge_1():
    reference = "первый второй третий четвёртый"
    reordered = "четвёртый третий второй первый"

    # Одинаковый набор слов -> ROUGE-1 не видит разницы, а ROUGE-L видит,
    # т.к. чувствителен к порядку (общая подпоследовательность короче).
    assert rouge_1_f1(reference, reordered) == 1.0
    assert rouge_l_f1(reference, reordered) < 1.0


def test_case_insensitive():
    assert rouge_1_f1("Общежитие", "общежитие") == 1.0
