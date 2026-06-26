from src.services.container import processor, document_manager, cache, html_loader, html_parser
from src.services.document.document_service import DocumentService
from src.services.document.clear_service import ClearService
from src.services.document.health_service import HealthService
from src.config import settings
from fastapi import HTTPException, Header

UPLOAD_DIR = "data/documents"

# Функции-провайдеры
# Создает и возвращает экзмепляр класса  
async def get_document_service() -> DocumentService:
    return DocumentService(processor, UPLOAD_DIR)

async def get_clear_service() -> ClearService:
    return ClearService(document_manager, cache)

async def get_health_service() -> HealthService:
    return HealthService(document_manager, cache)

# Защита admin-эндпоинтов
async def verify_admin_token(admin_token: str = Header(..., alias="X-Admin-Token")):
    """Проверяет admin токен"""
    if admin_token != settings.ADMIN_SECRET_TOKEN:
        raise HTTPException(
            status_code=403, 
            detail="Invalid or missing admin token"
        )
    return True