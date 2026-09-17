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
