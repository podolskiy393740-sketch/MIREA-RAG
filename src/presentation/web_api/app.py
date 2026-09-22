from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.infrastructure.embeddings.qwen_local_embedder import QwenLocalEmbedder
from src.infrastructure.llm.openai_llm import OpenAIClient, OpenAISettings
from src.infrastructure.llm.openrouter_llm import OpenRouterLLMClient, OpenRouterSettings
from src.presentation.web_api.router import api_router

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.embedder = QwenLocalEmbedder()

    llm = None
    try:
        s = OpenAISettings()
        if s.openai_api_key:
            llm = OpenAIClient(api_key=s.openai_api_key, model=s.openai_model)
    except ValidationError:
        pass

    if llm is None:
        try:
            s = OpenRouterSettings()
            if s.openrouter_api_key:
                llm = OpenRouterLLMClient(api_key=s.openrouter_api_key, model=s.openrouter_model)
        except ValidationError:
            pass

    app.state.llm = llm
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="МИРЭА RAG API", lifespan=_lifespan)
    app.include_router(api_router, prefix="/api")
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    return app


app = create_app()
