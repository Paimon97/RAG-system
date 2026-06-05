from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models as rest
from elasticsearch import AsyncElasticsearch

from src.config import settings

import uuid
import asyncio
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
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
                                "text": {
                                    "type": "text",
                                    "analyzer": "standard"
                                },
                                "metadata": {
                                    "type": "object"
                                },
                                "doc_id": {
                                    "type": "keyword"
                                }
                            }
                        }
                    }
                )
                logger.info(f"Created Elasticsearch index: {settings.ES_INDEX}")
            else:
                logger.info(f"Elasticsearch index already exists: {settings.ES_INDEX}")
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
    
    # Гибридный поиск: семантический + лексический
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        
        # Семантический поиск в Qdrant
        vector_results = await self._vector_search(query, top_k)
        
        # Лексический поиск в Elasticsearch
        lexical_results = await self._lexical_search(query, top_k)
        
        # Слияние результатов
        merged = self._fusion_results(vector_results, lexical_results, top_k)
        
        return merged
    
    # Семантический поиск в Qdrant
    # Добавить try except
    async def _vector_search(self, query: str, top_k: int) -> List[Dict]:

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
 
    # Лексический поиск в Elasticsearch
    async def _lexical_search(self, query: str, top_k: int) -> List[Dict]:
        try:
            # Проверяем существование индекса
            # Если нет возвращаем пустой массив
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
    
    # Слияние результатов с Reciprocal Rank Fusion
    def _fusion_results(self, vector_results: List[Dict], lexical_results: List[Dict], top_k: int) -> List[Dict]:
        
        # Если нет результатов, возвращаем пустой список
        if not vector_results and not lexical_results:
            return []
        
        # Словарь для хранения объединенных результатов
        merged_dict = {}
        
        # Обрабатываем семантические результаты
        for rank, result in enumerate(vector_results):
            # Создаем ключ из текста (первые 100 символов)
            key = result["text"][:1000] if result["text"] else str(rank)
            
            if key not in merged_dict:
                merged_dict[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 1.0 / (60 + rank + 1),  # RRF формула
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
    
    # Добавление документа в обе базы данных
    async def add_document(self, chunks: List[str], metadata: List[Dict]) -> str:
        try:
            doc_id = str(uuid.uuid4())
            
            # Добавляем в Qdrant
            await self._add_to_qdrant(chunks, metadata, doc_id)
            
            # Добавляем в Elasticsearch
            await self._add_to_elasticsearch(chunks, metadata, doc_id)
            
            logger.info(f"Document {doc_id} added successfully with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    # """Добавление чанков в Qdrant"""
    async def _add_to_qdrant(self, chunks: List[str], metadata: List[Dict], doc_id: str):
        try:
            # Получаем эмбеддинги для всех чанков
            embeddings = self.embedder.encode_batch(chunks)

            # Создаем точки для вставки
            points = []
            for i, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "text": chunk,
                        "metadata": meta
                    }
                )
                points.append(point)
            
            # Вставляем точки в коллекцию
            if points:
                await self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Added {len(points)} points to Qdrant")
                
        except Exception as e:
            logger.error(f"Error adding to Qdrant: {e}")
            raise
    
    # """Добавление чанков в Elasticsearch"""
    async def _add_to_elasticsearch(self, chunks: List[str], metadata: List[Dict], doc_id: str):
        try:
            for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
                doc = {
                    "text": chunk,
                    "metadata": meta,
                    "doc_id": doc_id,
                    "chunk_index": i
                }
                
                await self.es.index(
                    index=settings.ES_INDEX,
                    body=doc
                )
            
            logger.info(f"Added {len(chunks)} documents to Elasticsearch")
                
        except Exception as e:
            logger.error(f"Error adding to Elasticsearch: {e}")
            raise
    
    async def delete_document(self, doc_id: str):
        try:
            # Удаление из Qdrant
            await self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                )
            )
            
            # Удаление из Elasticsearch
            await self.es.delete_by_query(
                index=settings.ES_INDEX,
                body={
                    "query": {
                        "term": {"doc_id": doc_id}
                    }
                }
            )
            
            logger.info(f"Document {doc_id} deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    async def get_collection_info(self) -> Dict:
        try:
            collection_info = await self.qdrant.get_collection(self.collection_name)
            
            es_exists = await self.es.indices.exists(index=settings.ES_INDEX)
            es_count = 0
            if es_exists:
                count_result = await self.es.count(index=settings.ES_INDEX)
                es_count = count_result["count"]
            
            return {
                "qdrant_points": collection_info.points_count if hasattr(collection_info, 'points_count') else 0,
                "elasticsearch_docs": es_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"qdrant_points": 0, "elasticsearch_docs": 0}