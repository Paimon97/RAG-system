import redis
import json
import logging
from src.config import settings

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis = None
    
    @property
    def is_available(self):
        """Проверка доступности кеша"""
        return self.redis is not None
    
    def initialize(self):  # Убрали async
        """Синхронная инициализация"""
        try:
            self.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Проверяем подключение
            self.redis.ping()
            logger.info(f"Redis connected to {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Caching disabled.")
            self.redis = None
        return self
    
    def close(self):  # Убрали async
        """Синхронное закрытие"""
        if self.redis:
            try:
                self.redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis: {e}")
    
    def get(self, key: str):  # Убрали async
        """Синхронное получение"""
        if not self.redis:
            return None
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value, ttl: int = None):  # Убрали async
        """Синхронное сохранение"""
        if not self.redis:
            return
        try:
            ttl = ttl or settings.CACHE_TTL
            # Преобразуем в dict если нужно
            if hasattr(value, 'dict'):
                value = value.dict()
            elif hasattr(value, 'model_dump'):
                value = value.model_dump()
            
            self.redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error(f"Redis set error: {e}")