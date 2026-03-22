from __future__ import annotations

from pydantic import BaseModel, Field


class SchemaConfig(BaseModel):
    fields: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    enable_redaction: bool = False
    redaction_command: str = ""
    redaction_config: str = ""


class ProviderConfig(BaseModel):
    type: str = "local_vlm"


class NanonetsConfig(BaseModel):
    api_key_env: str = "NANONETS_API_KEY"
    endpoint: str | None = None
    low_confidence_threshold: float = 0.8  # fields scored below this trigger needs_review


class LocalVLMConfig(BaseModel):
    endpoint: str = "http://127.0.0.1:8080/extract"
    timeout_s: int = 120


class AppConfig(BaseModel):
    version: int = 1
    schema: SchemaConfig
    pipeline: PipelineConfig
    provider: ProviderConfig
    nanonets: NanonetsConfig = NanonetsConfig()
    local_vlm: LocalVLMConfig = LocalVLMConfig()

