import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    env: str = os.getenv("ENV", "dev")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/dev.db")
    ai_mode: str = os.getenv("AI_MODE", "rule")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    max_steps_default: int = int(os.getenv("MAX_STEPS_DEFAULT", "30"))
    tick_interval_ms: int = int(os.getenv("TICK_INTERVAL_MS", "800"))
    cors_origins: list[str] = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if x.strip()]


settings = Settings()
