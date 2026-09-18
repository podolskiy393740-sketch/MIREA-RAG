from src.domain.entities import Document, DocumentType
from src.infrastructure.chunking.structure_aware_chunker import StructureAwareChunker
from src.infrastructure.ingestion.html_parser import html_to_markdown

_MIREA_PAGE_HTML = """
<html>
  <body>
    <h1>Положение о военной кафедре</h1>
    <h2>Общие сведения</h2>
    <p>Военная кафедра проводит обучение студентов по программе подготовки офицеров запаса.</p>
    <h2>Порядок зачисления</h2>
    <p>Зачисление проводится на конкурсной основе по результатам медицинской комиссии.</p>
    <script>console.log("удалить");</script>
  </body>
</html>
"""


def test_headings_become_atx_markdown():
    markdown = html_to_markdown(_MIREA_PAGE_HTML)

    assert "# Положение о военной кафедре" in markdown
    assert "## Порядок зачисления" in markdown


def test_script_tags_are_stripped():
    markdown = html_to_markdown(_MIREA_PAGE_HTML)

    assert "console.log" not in markdown


def test_parsed_html_is_chunkable_with_heading_context():
    """Связка html_parser -> StructureAwareChunker: именно ради этого
    формата (ATX-заголовки) парсер и написан так, а не иначе."""
    markdown = html_to_markdown(_MIREA_PAGE_HTML)
    document = Document(
        id="doc-military", source_url="https://mirea.ru/military", doc_type=DocumentType.STRUCTURED_HTML,
        raw_text=markdown,
    )

    chunks = StructureAwareChunker(max_chars=1000).chunk(document)

    assert len(chunks) == 2
    assert "Порядок зачисления" in chunks[1].text
    assert "Положение о военной кафедре" in chunks[1].text


# Упрощённая копия реального шаблона mirea.ru (см. data/raw_html/*.html,
# получены через Wayback Machine): мега-меню на сотни пунктов + заголовок
# и текст статьи в отдельных классах page-title/app-content.
_REAL_TEMPLATE_HTML = """
<html>
  <body>
    <nav class="top_menu__content">
      <ul>
        <li><a href="/abitur/">Абитуриентам</a></li>
        <li><a href="/about/history/">История вуза</a>
          <ul>
            <li><a href="/about/history/1997/">1997</a></li>
            <li><a href="/about/history/1998/">1998</a></li>
          </ul>
        </li>
      </ul>
    </nav>
    <h1 class="uk-heading-bullet page-title">Общежития</h1>
    <div class="uk-width-1-1 app-content">
      <p>Студенческий городок РТУ МИРЭА — комплекс из шести корпусов.</p>
      <h2>Общежитие №1</h2>
      <p>Комнаты подготовки к занятиям, спортзал.</p>
    </div>
    <footer>
      <a href="/about/">Об Университете</a>
      <a href="/about/history/">История вуза</a>
    </footer>
  </body>
</html>
"""


def test_extracts_only_title_and_content_ignoring_mega_menu():
    markdown = html_to_markdown(_REAL_TEMPLATE_HTML)

    assert "# Общежития" in markdown
    assert "## Общежитие №1" in markdown
    assert "Абитуриентам" not in markdown
    assert "История вуза" not in markdown


def test_falls_back_to_whole_body_without_known_content_class():
    """_MIREA_PAGE_HTML не использует шаблон mirea.ru (нет app-content) —
    парсер не должен падать, а должен отдать всё тело как раньше."""
    markdown = html_to_markdown(_MIREA_PAGE_HTML)

    assert "Положение о военной кафедре" in markdown


# Более старый шаблон /ads/ (например, страницы 2025 года про поступление
# в военный учебный центр) — заголовок без класса, текст в news-item-text.
_NEWS_ITEM_TEMPLATE_HTML = """
<html>
  <body>
    <nav class="top_menu__content">
      <ul><li><a href="/abitur/">Абитуриентам</a></li></ul>
    </nav>
    <h1>Поступление в Военный учебный центр при РТУ МИРЭА</h1>
    <div class="uk-margin-bottom">05.03.2025</div>
    <div class="news-item-text uk-margin-bottom">
      <p>Горячая линия по вопросам поступления: 7 499 600-80-80, доб. 32604.</p>
    </div>
    <footer><a href="/about/">Об Университете</a></footer>
  </body>
</html>
"""


def test_extracts_news_item_template_ignoring_mega_menu():
    markdown = html_to_markdown(_NEWS_ITEM_TEMPLATE_HTML)

    assert "# Поступление в Военный учебный центр при РТУ МИРЭА" in markdown
    assert "Горячая линия" in markdown
    assert "Абитуриентам" not in markdown
    assert "Об Университете" not in markdown
