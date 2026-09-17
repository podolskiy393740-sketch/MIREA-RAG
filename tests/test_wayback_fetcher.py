import httpx
import pytest

from src.infrastructure.ingestion.wayback_fetcher import fetch_snapshot_html, find_latest_snapshot, Snapshot


@pytest.mark.asyncio
async def test_find_latest_snapshot_picks_last_row():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[["timestamp"], ["20200101000000"], ["20230601000000"]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await find_latest_snapshot(client, "https://www.mirea.ru/example/")

    assert snapshot is not None
    assert snapshot.timestamp == "20230601000000"


@pytest.mark.asyncio
async def test_find_latest_snapshot_retries_on_invalid_json():
    """Архив иногда отдаёт HTML-страницу "Temporarily Offline" вместо JSON
    с кодом 200 — это должно вести к ретраю, а не к падению."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, text="<html>Temporarily Offline</html>")
        return httpx.Response(200, json=[["timestamp"], ["20230601000000"]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await find_latest_snapshot(client, "https://www.mirea.ru/example/", backoff_seconds=0)

    assert snapshot is not None
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_find_latest_snapshot_returns_none_when_no_rows():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[["timestamp"]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await find_latest_snapshot(client, "https://www.mirea.ru/missing/")

    assert snapshot is None


@pytest.mark.asyncio
async def test_fetch_snapshot_html_returns_body_on_success():
    body = "<html>" + "содержимое страницы " * 100 + "</html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    snapshot = Snapshot(original_url="https://www.mirea.ru/example/", timestamp="20230601000000")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        html = await fetch_snapshot_html(client, snapshot, retries=1)

    assert html == body


@pytest.mark.asyncio
async def test_fetch_snapshot_html_retries_on_short_response():
    calls = {"count": 0}
    body = "<html>" + "содержимое страницы " * 100 + "</html>"

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, text="слишком коротко")
        return httpx.Response(200, text=body)

    snapshot = Snapshot(original_url="https://www.mirea.ru/example/", timestamp="20230601000000")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        html = await fetch_snapshot_html(client, snapshot, retries=3, backoff_seconds=0)

    assert html == body
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_fetch_snapshot_html_raises_after_exhausting_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="всегда коротко")

    snapshot = Snapshot(original_url="https://www.mirea.ru/example/", timestamp="20230601000000")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError):
            await fetch_snapshot_html(client, snapshot, retries=1)
