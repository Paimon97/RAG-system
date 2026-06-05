import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "documents"
    VECTOR_SIZE: int = 1024
    
    # Elasticsearch
    ES_HOST: str = "http://localhost:9200"
    ES_INDEX: str = "documents"
    
    # Redis кеш
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    CACHE_TTL: int = 3600
    
    # Модели
    EMBEDDING_MODEL: str = "ai-forever/ru-en-RoSBERTa"
    LLM_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct" #"microsoft/DialoGPT-small"
    # SUMMARIZATION_MODEL: str = "IlyaGusev/rut5_base_sum_gazeta"

    
    # Параметры обработки
    CHUNK_SIZE: int = 400  # 
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 3
    CONFIDENCE_THRESHOLD: float = 0.8
    MAX_SEQUENCE_LENGTH: int = 216  # Явное ограничение
    SUMMARIZATION_MAX_INPUT_TOKENS: int = 1024
    
    class Config:
        env_file = ".env"

settings = Settings()