from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import SecretStr
from typing import List
import os
from enum import Enum

class LLMType(str, Enum):
    OLLAMA = "OLLAMA"
    OPENAI = "OPENAI"
    DEEPSEEK = "DEEPSEEK"

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
    HISTORY_ITEMS: int = 3 # limits the number of newest history item to be used as context
    RAG_DIR: Path = "/home/devmiftahul/nlp/repositories/agentic_with_mcp/tool/rag"
    SYSTEM_PROMPT_PATH: Path = "/home/devmiftahul/nlp/llm_dev/api_openai/prompt/system_comment.txt"
    MCP_SERVERS_CONFIG_PATH: Path = "/home/devmiftahul/nlp/repositories/agentic_with_mcp/servers.json"

    LLM_TYPE: LLMType = LLMType.OPENAI
    OPENAI_API_KEY: SecretStr
    DEEPSEEK_API_KEY: SecretStr

    # caching env
    REDIS_URL: str = "redis://:tuntun123@localhost:6379"
    TIMEBOUND_TOOLS: List[str] = ["stock_price", "combined_bvhl_pricemod"]

    # tuntun backend endpoints
    BE_COMPANY_QUALITY_URL: str = "http://10.192.1.228:8083/api/v1/tuntun-guidance/company-quality"
    BE_FAIR_VALUE_URL: str = "http://10.192.1.228:8083/api/v1/tuntun-guidance/fair-value"
    BE_ORDERBOOK_HEADER_URL: str = "http://10.192.1.245:8080/orderbook/header"
    BE_POSITIVE_SIGNAL_URL: str = "http://10.192.1.228:8083/backend/stock/get-positive-signals"
    BE_PREV_TRADING_DATE_URL: str = "http://10.192.1.245:8080/market-status/working-day/by-days"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Config()  # type: ignore

# __import__("pprint").pprint(settings.__dict__)
