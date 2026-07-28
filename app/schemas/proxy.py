from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionRequest(BaseModel):

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., min_length=1)
    messages: list[dict[str, Any]] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | str | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None

class ModelInfo(BaseModel):

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "ollama"

class ModelsResponse(BaseModel):

    object: str = "list"
    data: list[ModelInfo]

class ModelTag(BaseModel):

    name: str
    model: str
    modified_at: str = ""
    size: int = 0
    digest: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

class TagsResponse(BaseModel):

    models: list[ModelTag]
