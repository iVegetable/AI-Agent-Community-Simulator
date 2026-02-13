"""Runtime configuration loaded from environment variables.

This module centralizes backend settings such as database URL, AI provider
mode, OpenAI model options, and API behavior defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Typed settings object used across the backend."""

    env: str = os.getenv("ENV", "dev")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/dev.db")
    ai_mode: str = os.getenv("AI_MODE", "rule").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_max_output_tokens: int = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "256"))
    openai_timeout_ms: int = int(os.getenv("OPENAI_TIMEOUT_MS", "30000"))
    openai_max_retries: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    openai_concurrency: int = int(os.getenv("OPENAI_CONCURRENCY", "3"))
    max_steps_default: int = int(os.getenv("MAX_STEPS_DEFAULT", "30"))
    tick_interval_ms: int = int(os.getenv("TICK_INTERVAL_MS", "800"))
    cors_origins: list[str] = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if x.strip()]


settings = Settings()
