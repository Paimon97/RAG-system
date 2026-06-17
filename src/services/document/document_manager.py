# src/services/retriever/document_manager.py
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from elasticsearch import AsyncElasticsearch

from src.config import settings

import uuid
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DocumentManager:
    """Класс для управления документами в Qdrant и Elasticsearch"""
    
    def __init__(self, qdrant_client: AsyncQdrantClient, es_client: AsyncElasticsearch, embedder, collection_name: str):
        self.qdrant = qdrant_client
        self.es = es_client
        self.embedder = embedder
        self.collection_name = collection_name

    async def add_document(self, chunks: List[str], metadata: List[Dict]) -> str:
        """Добавление документа в обе базы данных"""
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

    async def _add_to_qdrant(self, chunks: List[str], metadata: List[Dict], doc_id: str):
        """Добавление чанков в Qdrant"""
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

    async def _add_to_elasticsearch(self, chunks: List[str], metadata: List[Dict], doc_id: str):
        """Добавление чанков в Elasticsearch"""
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
        """Удаление документа из обеих баз данных"""
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
        """Получение информации о коллекциях"""
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