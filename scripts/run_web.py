"""Запуск веб-сервера локально или на VPS.

Использование:
    python -m scripts.run_web           # localhost:8000
    python -m scripts.run_web --port 80 # нестандартный порт
"""
import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("src.presentation.web_api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
