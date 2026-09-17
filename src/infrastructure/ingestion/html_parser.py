from __future__ import annotations

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

# Шаблон сайта mirea.ru (проверено на реальных страницах через Wayback
# Machine, см. docs/data-collection.md): заголовок статьи и её текст лежат
# в этих классах, а не во всём <body> — там ещё мега-меню с сотнями
# пунктов навигации на каждую страницу, которое иначе тоже попадёт в текст
# для чанкинга и забьёт индекс мусором вместо контента.
_CONTENT_CLASS = "app-content"
_TITLE_CLASS = "page-title"


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

    main = _extract_main_content(soup)
    return markdownify(str(main), heading_style="ATX").strip()


def _extract_main_content(soup: BeautifulSoup) -> Tag:
    content = soup.find(class_=_CONTENT_CLASS)
    if content is None:
        # Незнакомый шаблон страницы (например, старый архивный снапшот) —
        # лучше отдать всё тело, чем упасть, но это нужно проверять глазами.
        return soup.body or soup

    title = soup.find(class_=_TITLE_CLASS)
    if title is None:
        return content

    wrapper = BeautifulSoup("<div></div>", "html.parser").div
    wrapper.append(title.extract())
    wrapper.append(content.extract())
    return wrapper
