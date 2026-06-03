from src.services.container import processor, retriever, cache
from src.services.document_service import DocumentService
from src.services.clear_service import ClearService
from src.services.health_service import HealthService
from src.config import settings
from fastapi import HTTPException, Header

UPLOAD_DIR = "data/documents"

# Функции-провайдеры
async def get_document_service() -> DocumentService:
    return DocumentService(processor, UPLOAD_DIR)

async def get_clear_service() -> ClearService:
    # Исправленный ClearService (без storage_service)
    return ClearService(retriever, cache)

async def get_health_service() -> HealthService:
    return HealthService(retriever, cache)

# Защита admin-эндпоинтов
# async def verify_admin_token(admin_token: str = Header(..., alias="X-Admin-Token")):
#     """Проверяет admin токен"""
#     if admin_token != settings.ADMIN_SECRET_TOKEN:
#         raise HTTPException(
#             status_code=403, 
#             detail="Invalid or missing admin token"
#         )
#     return True