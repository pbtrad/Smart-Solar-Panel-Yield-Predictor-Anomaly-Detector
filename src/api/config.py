from pydantic_settings import BaseSettings
import os


class APISettings(BaseSettings):
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY", "test-secret-key"
    )  # Need to change with real one
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Smart Solar API"
    VERSION: str = "1.0.0"

    RATE_LIMIT_PER_MINUTE: int = 60

    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]

    MODEL_PATH: str = "models/"
    DEFAULT_FORECAST_PERIODS: int = 24

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = APISettings()
