from src.config import settings

class ClearService:

    def __init__(self, retriever, redis_client):
        self.retriever = retriever
        # self.storage = storage_service
        self.redis = redis_client

    async def clear_all(self):
        # 1. Очистка Elasticsearch
        try:
            # Проверяем существует ли индекс
            if await self.retriever.es.indices.exists(index=settings.ES_INDEX):
                # Удаляем индекс
                await self.retriever.es.indices.delete(index=settings.ES_INDEX)

            
            # Создаем индекс заново
            await self.retriever.es.indices.create(
                index=settings.ES_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "text": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "metadata": {"type": "object"},
                            "doc_id": {"type": "keyword"}
                        }
                    }
                }
            )

            es_cleaned = True
        except Exception as e:
            es_cleaned = False
        
        # 2. Очистка Qdrant
        try:
            # Проверяем существует ли коллекция
            collections = await self.retriever.qdrant.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if settings.QDRANT_COLLECTION in collection_names:
                # Удаляем коллекцию
                await self.retriever.qdrant.delete_collection(collection_name=settings.QDRANT_COLLECTION)
            
            # Создаем коллекцию заново
            from qdrant_client.models import Distance, VectorParams
            
            await self.retriever.qdrant.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )

            qdrant_cleaned = True
        except Exception as e:
            qdrant_cleaned = False
        
        # 4. Очистка загруженных файлов

        try:
            import os
            docs_dir = "data/documents"
            files_removed = 0
            if os.path.exists(docs_dir):
                for file in os.listdir(docs_dir):
                    file_path = os.path.join(docs_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        files_removed += 1

        except Exception as e:
            files_removed = 0
        
        return {
            "status": "success",
            "message": "Все данные очищены",
            "details": {
                "elasticsearch": "очищен" if es_cleaned else "ошибка",
                "qdrant": "очищен" if qdrant_cleaned else "ошибка",
                "files_removed": files_removed
            }
        }
        
    async def clear_cache(self):
        try:
            # Проверяем доступность кэша
            if not self.redis:
                return {
                    "status": "error",
                    "message": "Кэш не доступен (Redis не подключен)"
                }
            
            # Получаем количество ключей до очистки
            keys_before = len(self.redis.keys('*'))
            
            # Очищаем кэш
            self.redis.flushdb()
            
            # Проверяем результат
            keys_after = len(self.redis.keys('*'))
            
            return {
                "status": "success",
                "message": "Кэш успешно очищен",
                "details": {
                    "keys_deleted": keys_before,
                    "keys_remaining": keys_after
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Ошибка очистки кэша: {str(e)}"
            }

    # try:
    #     # 1. Очистка Elasticsearch
    #     try:
    #         # Проверяем существует ли индекс
    #         if await retriever.es.indices.exists(index=settings.ES_INDEX):
    #             # Удаляем индекс
    #             await retriever.es.indices.delete(index=settings.ES_INDEX)

            
    #         # Создаем индекс заново
    #         await retriever.es.indices.create(
    #             index=settings.ES_INDEX,
    #             body={
    #                 "mappings": {
    #                     "properties": {
    #                         "text": {
    #                             "type": "text",
    #                             "analyzer": "standard"
    #                         },
    #                         "metadata": {"type": "object"},
    #                         "doc_id": {"type": "keyword"}
    #                     }
    #                 }
    #             }
    #         )
    #         logger.info(f"Индекс {settings.ES_INDEX} создан заново")
    #         es_cleaned = True
    #     except Exception as e:
    #         logger.error(f"Ошибка очистки Elasticsearch: {e}")
    #         es_cleaned = False
        
    #     # 2. Очистка Qdrant
    #     logger.info("Очистка Qdrant...")
    #     try:
    #         # Проверяем существует ли коллекция
    #         collections = await retriever.qdrant.get_collections()
    #         collection_names = [c.name for c in collections.collections]
            
    #         if settings.QDRANT_COLLECTION in collection_names:
    #             # Удаляем коллекцию
    #             await retriever.qdrant.delete_collection(collection_name=settings.QDRANT_COLLECTION)
    #             logger.info(f"Коллекция {settings.QDRANT_COLLECTION} удалена")
            
    #         # Создаем коллекцию заново
    #         from qdrant_client.models import Distance, VectorParams
            
    #         await retriever.qdrant.create_collection(
    #             collection_name=settings.QDRANT_COLLECTION,
    #             vectors_config=VectorParams(
    #                 size=settings.VECTOR_SIZE,
    #                 distance=Distance.COSINE
    #             )
    #         )
    #         logger.info(f"Коллекция {settings.QDRANT_COLLECTION} создана заново")
    #         qdrant_cleaned = True
    #     except Exception as e:
    #         logger.error(f"Ошибка очистки Qdrant: {e}")
    #         qdrant_cleaned = False
        
    #     # 4. Очистка загруженных файлов
    #     logger.info("Очистка файлов...")
    #     try:
    #         import os
    #         docs_dir = "data/documents"
    #         files_removed = 0
    #         if os.path.exists(docs_dir):
    #             for file in os.listdir(docs_dir):
    #                 file_path = os.path.join(docs_dir, file)
    #                 if os.path.isfile(file_path):
    #                     os.remove(file_path)
    #                     files_removed += 1
    #         logger.info(f"Удалено файлов: {files_removed}")
    #     except Exception as e:
    #         logger.error(f"Ошибка очистки файлов: {e}")
    #         files_removed = 0
        
    #     return {
    #         "status": "success",
    #         "message": "Все данные очищены",
    #         "details": {
    #             "elasticsearch": "очищен" if es_cleaned else "ошибка",
    #             "qdrant": "очищен" if qdrant_cleaned else "ошибка",
    #             "files_removed": files_removed
    #         }
    #     }
        
    # except Exception as e:
    #     logger.error(f"Критическая ошибка при очистке: {e}")
    #     raise HTTPException(status_code=500, detail=f"Ошибка очистки: {str(e)}")