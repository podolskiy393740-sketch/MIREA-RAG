from __future__ import annotations

from bs4 import BeautifulSoup
from markdownify import markdownify


def html_to_markdown(html: str) -> str:
    """HTML страницы МИРЭА -> markdown с ATX-заголовками ('#'..'######').

    Формат согласован со StructureAwareChunker (см.
    src/infrastructure/chunking/structure_aware_chunker.py) — он парсит
    именно такие заголовки, чтобы нести путь разделов как контекст чанка.

    markdownify.strip=[...] только убирает разметку тега, но не его
    текст (например, у <script> внутри остался бы JS-код как plain text) —
    поэтому script/style вырезаются целиком через BeautifulSoup заранее.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return markdownify(str(soup), heading_style="ATX").strip()
