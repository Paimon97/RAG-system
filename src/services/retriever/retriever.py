# src/services/retriever/retriever.py
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from elasticsearch import AsyncElasticsearch

from src.config import settings

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Гибридный поиск: семантический + лексический"""
    
    def __init__(self, embedder):
        self.qdrant = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.es = AsyncElasticsearch(settings.ES_HOST)
        self.embedder = embedder
        self.collection_name = settings.QDRANT_COLLECTION

    async def close(self):
        """Закрытие соединений"""
        try:
            logger.info("Закрытие соединений")
            if hasattr(self.qdrant, 'close'):
                await self.qdrant.close()
            if hasattr(self.es, 'close'):
                await self.es.close()
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединений: {e}")

    async def initialize(self):
        """Инициализация коллекций и индексов"""
        try:
            # Проверка существования коллекции в Qdrant
            collections = await self.qdrant.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                await self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")
            
            # Проверка индекса в Elasticsearch
            if not await self.es.indices.exists(index=settings.ES_INDEX):
                await self.es.indices.create(
                    index=settings.ES_INDEX,
                    body={
                        "mappings": {
                            "properties": {
                                "text": {"type": "text", "analyzer": "standard"},
                                "metadata": {"type": "object"},
                                "doc_id": {"type": "keyword"}
                            }
                        }
                    }
                )
                logger.info(f"Created Elasticsearch index: {settings.ES_INDEX}")
            else:
                logger.info(f"Elasticsearch index already exists: {settings.ES_INDEX}")
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Гибридный поиск: семантический + лексический"""
        # Семантический поиск в Qdrant
        vector_results = await self._vector_search(query, top_k)
        
        # Лексический поиск в Elasticsearch
        lexical_results = await self._lexical_search(query, top_k)
        
        # Слияние результатов
        merged = self._fusion_results(vector_results, lexical_results, top_k)
        
        return merged

    async def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Семантический поиск в Qdrant"""
        try:
            # Получаем вектор запроса
            query_vector = self.embedder.encode_batch([query], is_query=True)
            
            # Используем метод query_points (новый API Qdrant)
            search_result = await self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector[0],
                limit=top_k,
                with_payload=True
            )
                
            # Форматируем результаты
            results = []
            for scored_point in search_result.points:
                if scored_point.payload:
                    results.append({
                        "text": scored_point.payload.get("text", ""),
                        "metadata": scored_point.payload.get("metadata", {}),
                        "score": scored_point.score,
                        "source": "semantic"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    async def _lexical_search(self, query: str, top_k: int) -> List[Dict]:
        """Лексический поиск в Elasticsearch"""
        try:
            # Проверяем существование индекса
            if not await self.es.indices.exists(index=settings.ES_INDEX):
                logger.warning(f"Elasticsearch index {settings.ES_INDEX} does not exist")
                return []
            
            # BM25 поиск
            response = await self.es.search(
                index=settings.ES_INDEX,
                body={
                    "query": {
                        "match": {
                            "text": {
                                "query": query,
                                "operator": "or"
                            }
                        }
                    },
                    "size": top_k
                }
            )
            
            # Форматируем результаты
            results = []
            for hit in response["hits"]["hits"]:
                source = hit.get("_source", {})
                results.append({
                    "text": source.get("text", ""),
                    "metadata": source.get("metadata", {}),
                    "score": hit["_score"],
                    "source": "lexical"
                })
            
            logger.info(f"Lexical search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Lexical search error: {e}")
            return []

    def _fusion_results(self, vector_results: List[Dict], lexical_results: List[Dict], top_k: int) -> List[Dict]:
        """Слияние результатов с Reciprocal Rank Fusion"""
        # Если нет результатов, возвращаем пустой список
        if not vector_results and not lexical_results:
            return []
        
        # Словарь для хранения объединенных результатов
        merged_dict = {}
        
        # Обрабатываем семантические результаты
        for rank, result in enumerate(vector_results):
            # Создаем ключ из текста (первые 1000 символов)
            key = result["text"][:1000] if result["text"] else str(rank)
            
            if key not in merged_dict:
                merged_dict[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 1.0 / (60 + rank + 1),
                    "semantic_score": result["score"],
                    "lexical_score": 0
                }
            else:
                merged_dict[key]["rrf_score"] += 1.0 / (60 + rank + 1)
                merged_dict[key]["semantic_score"] = result["score"]
        
        # Обрабатываем лексические результаты
        for rank, result in enumerate(lexical_results):
            key = result["text"][:1000] if result["text"] else str(rank)
            
            if key not in merged_dict:
                merged_dict[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 1.0 / (60 + rank + 1),
                    "semantic_score": 0,
                    "lexical_score": result["score"]
                }
            else:
                merged_dict[key]["rrf_score"] += 1.0 / (60 + rank + 1)
                merged_dict[key]["lexical_score"] = result["score"]
        
        # Сортируем по RRF score и берем top_k
        sorted_results = sorted(
            merged_dict.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        return sorted_results[:top_k]