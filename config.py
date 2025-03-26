from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import SecretStr
from typing import List
import os

class Config(BaseSettings):

    LOG_LEVEL: str = "INFO"

    LOG_DB_URL: str = "http://localhost:8529"
    LOG_DB_USERNAME: str = "root"
    LOG_DB_PASSWORD: SecretStr = "tuntun123"
    LOG_DB_NAME: str = "tuntun_chatbot"
    LOG_DB_COLLECTION_NAME: str = "chat_logs"

    API_KEY: SecretStr = "ac7c07ad4851146d36ba0af67ad8bfb5f945c694f122a0babb14ff2632b60196"
    API_VERSION: str = "0.1.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    PROMPT_VERSION: str = "v2"

    # caching env
    REDIS_URL: str = "redis://:tuntun123@localhost:6379"
    SKIP_TOOLS: List[str] = ["stock_price", "combined_bvhl_pricemod"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Config()  # type: ignore

__import__("pprint").pprint(settings.__dict__)
