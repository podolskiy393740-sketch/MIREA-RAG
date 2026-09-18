from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def rouge_1_f1(reference: str, prediction: str) -> float:
    """Пересечение множества слов (с повторами) между эталоном и ответом
    модели — грубая, но интерпретируемая метрика полноты/точности по
    словам. Референс методологии — eval-подход chernenko-s/mirea-rag,
    на который прямо ссылался куратор (см. CLAUDE.md)."""
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    overlap = sum((Counter(ref_tokens) & Counter(pred_tokens)).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return _f1(precision, recall)


def rouge_l_f1(reference: str, prediction: str) -> float:
    """F1 по наибольшей общей подпоследовательности слов — в отличие от
    ROUGE-1, чувствительна к порядку слов."""
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return _f1(precision, recall)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]
