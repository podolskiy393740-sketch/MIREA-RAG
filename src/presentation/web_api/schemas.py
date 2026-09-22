from __future__ import annotations

from pydantic import BaseModel


class UserContextRequest(BaseModel):
    course: int | None = None
    faculty: str | None = None


class AskRequest(BaseModel):
    question: str
    user_context: UserContextRequest | None = None


class AskResponse(BaseModel):
    answer: str
    needs_human_fallback: bool
