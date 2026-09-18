"""Скачивает реальные страницы mirea.ru через Wayback Machine.

Почему не напрямую с mirea.ru: сайт закрыт DDoS-Guard по гео-блокировке
для инфраструктуры, где живёт этот код (см. docs/data-collection.md).
web.archive.org отдаёт кэш публично опубликованных страниц с отдельного
сервера — это не обход блокировки mirea.ru, а чтение архива, к которому
у mirea.ru нет отношения.

Использование:
    python -m scripts.fetch_real_pages --out data/raw_html
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from src.infrastructure.ingestion.wayback_fetcher import fetch_snapshot_html, find_latest_snapshot

# Курируемый список страниц, релевантных первокурснику (курсовая — версия
# для первокурсников, см. CLAUDE.md). Добавлять новые страницы сюда же.
TARGET_PAGES = {
    "obshchezhitiya": "https://www.mirea.ru/about/infrastructure/obshchezhitiya/",
    "stolovye_bufety": "https://www.mirea.ru/about/infrastructure/stolovye-bufety/",
    "sportivnaya_infrastruktura": "https://www.mirea.ru/about/infrastructure/sportivnaya-infrastruktura/",
    "start_uchebnogo_goda_pervokursniki": "https://www.mirea.ru/ads/o-nachale-uchebnogo-goda-dlya-pervokursnikov-rtu-mirea/",
    # Старые /education/military-training/ и .../military-department/
    # существуют в архиве только в шаблоне 2015-2020 годов (парсер их не
    # разбирает избирательно, падает на fallback). Эта страница из /ads/
    # 2025 года — рабочая замена с тем же смыслом (шаблон "news-item-text",
    # см. html_parser.py).
    "voenniy_uchebniy_tsentr": (
        "https://www.mirea.ru/ads/postuplenie-v-voennyy-uchebnyy-tsentr-pri-rtu-mirea-otvety-na-voprosy-i-kontaktnaya-informatsiya/"
    ),
}


async def fetch_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    # Мержим с уже сохранённым манифестом, а не перезаписываем целиком —
    # архив периодически недоступен для части (или всех) страниц за один
    # прогон, и предыдущие успешные записи не должны из-за этого теряться.
    manifest: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
            manifest[entry["name"]] = entry

    async with httpx.AsyncClient() as client:
        for name, url in TARGET_PAGES.items():
            try:
                snapshot = await find_latest_snapshot(client, url)
                if snapshot is None:
                    print(f"{name}: снапшот не найден, пропускаю")
                    continue

                html = await fetch_snapshot_html(client, snapshot)
            except RuntimeError as exc:
                # Архив может быть недоступен для конкретной страницы прямо
                # сейчас — не роняем сбор остальных страниц из-за одной.
                print(f"{name}: не удалось скачать ({exc}), пропускаю")
                continue

            (out_dir / f"{name}.html").write_text(html, encoding="utf-8")
            manifest[name] = {
                "name": name,
                "source_url": url,
                "wayback_timestamp": snapshot.timestamp,
                "raw_url": snapshot.raw_url,
            }
            print(f"{name}: OK ({len(html)} байт, снапшот {snapshot.timestamp})")

            await asyncio.sleep(3)  # вежливая пауза между запросами к архиву

    manifest_path.write_text(
        json.dumps(list(manifest.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw_html", help="Каталог для сохранения HTML")
    args = parser.parse_args()

    asyncio.run(fetch_all(Path(args.out)))
