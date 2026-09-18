from __future__ import annotations

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

# mirea.ru использует не один шаблон, а несколько (менялись со временем) —
# заголовок статьи и её текст лежат в этих контейнерах, а не во всём
# <body>, где ещё мега-меню с сотнями пунктов навигации на каждую страницу,
# которое иначе тоже попадёт в текст для чанкинга и забьёт индекс мусором.
# Проверено на реальных страницах через Wayback Machine — см.
# docs/data-collection.md. Каждый новый обнаруженный шаблон — новая функция
# в _EXTRACTION_STRATEGIES, пробуются по очереди, первое совпадение побеждает.


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
    for strategy in _EXTRACTION_STRATEGIES:
        extracted = strategy(soup)
        if extracted is not None:
            return extracted
    # Незнакомый шаблон страницы (например, старый архивный снапшот) —
    # лучше отдать всё тело, чем упасть, но это нужно проверять глазами.
    return soup.body or soup


def _extract_app_content_template(soup: BeautifulSoup) -> Tag | None:
    """Основной шаблон разделов сайта (/about/, /education/ и т.п.)."""
    content = soup.find(class_="app-content")
    if content is None:
        return None
    return _combine(soup.find(class_="page-title"), content)


def _extract_news_item_template(soup: BeautifulSoup) -> Tag | None:
    """Более старый шаблон объявлений (/ads/) без класса на заголовке —
    например, страницы 2025 года про поступление в военный учебный центр."""
    content = soup.find(class_="news-item-text")
    if content is None:
        return None
    return _combine(soup.find("h1"), content)


_EXTRACTION_STRATEGIES = (_extract_app_content_template, _extract_news_item_template)


def _combine(title: Tag | None, content: Tag) -> Tag:
    if title is None:
        return content
    wrapper = BeautifulSoup("<div></div>", "html.parser").div
    wrapper.append(title.extract())
    wrapper.append(content.extract())
    return wrapper
