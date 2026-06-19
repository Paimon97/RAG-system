from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends
from src.services.document.clear_service import ClearService
from src.services.document.health_service import HealthService
from src.services.document.document_service import DocumentService
from src.api.dependencies import get_document_service, get_clear_service, get_health_service
from src.models.schemas import QueryRequest, UploadResponse, QueryResponse
from src.services.container import orchestrator

import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Принимает ответ пользователя и возвращает ответ на основе документа.
@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):  
    return await orchestrator.answer(
        question=request.question,
        top_k=request.top_k
    )

# Загрузка страницы сразу из WIKI
@router.post("/wiki/import")
async def import_wiki(
    url: str,
    document_service: DocumentService = Depends(get_document_service)
):
    return await document_service.import_wiki(url)

# Загрузка нового txt документа для индексации.
@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service)
    ):
    # Сохарняет файл на диск. Возвращает информацию о сохраненом файле пути.
    uploaded = await document_service.upload_document(file)
    # Добавляет асинхронную фоновую задачу для обработки документа.
    # Обработка выполняется после отправки ответа.
    background_tasks.add_task(document_service.process_document, uploaded["path"])

    return UploadResponse(
        filename=uploaded["filename"],
        status="processing"
    )

# Проверка работоспособности сервисов
@router.get("/health")
async def health_check(health_service: HealthService = Depends(get_health_service)):
    return await health_service.get_health()

# Полная очистка всех данных в Elasticsearch и Qdrant
@router.post("/admin/clear-all")
async def clear_all_data(clear_service: ClearService = Depends(get_clear_service)): # _: bool = Depends(verify_admin_token),
    logger.warning("Запущена полная очистка баз данных")
    return await clear_service.clear_all()

# Очистка только кэша (Redis), без удаления документов
@router.post("/cache/clear")
async def clear_cache(clear_service: ClearService = Depends(get_clear_service)):
    logger.warning("Запущена очистка кэша")
    return await clear_service.clear_cache()

# @router.get("/cache/stats")
# async def cache_stats():
#     """Получение статистики кэша"""
#     try:
#         if not cache.redis:
#             return {
#                 "status": "error",
#                 "message": "Кэш не доступен"
#             }
        
#         # Получаем все ключи
#         keys = cache.redis.keys('*')
        
#         # Получаем информацию о каждом ключе (опционально)
#         key_info = []
#         for key in keys[:10]:  # Показываем только первые 10 ключей
#             ttl = cache.redis.ttl(key)
#             key_info.append({
#                 "key": key,
#                 "ttl_seconds": ttl if ttl > 0 else "persistent"
#             })
        
#         return {
#             "status": "success",
#             "stats": {
#                 "total_keys": len(keys),
#                 "redis_available": True,
#                 "sample_keys": key_info
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Ошибка получения статистики кэша: {e}")
#         return {
#             "status": "error",
#             "message": str(e)
#         }
    
    
# Удаление документа по ID
# @app.delete("/documents/{doc_id}")
# async def delete_document(doc_id: str):
#     try:
#         retriever.delete_document(doc_id)
#         return {"status": "deleted", "doc_id": doc_id}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
