# test_generator.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from src.services.generator.llm_generator import SafeLLMGenerator
from src.services.generator.generator_service import GenerationService
from src.services.generator.promt_builder import PromptBuilder
from src.services.retriever.retriever import HybridRetriever
from src.services.embedder import EmbeddingService
from concurrent.futures import ThreadPoolExecutor

async def test_generator_with_qdrant():
    # 1. Создаем компоненты
    embedder = EmbeddingService()
    retriever = HybridRetriever(embedder=embedder)
    generator = SafeLLMGenerator()
    prompt_builder = PromptBuilder()
    thread_pool = ThreadPoolExecutor(max_workers=1)
    
    generation_service = GenerationService(
        llm_generator=generator,
        prompt_builder=prompt_builder,
        thread_pool=thread_pool
    )
    
    # 2. Вопрос пользователя
    query = "Где можно прочитать лицензионное соглашение?"
    
    # 3. Ищем контекст в Qdrant (используем метод search)
    print(f"🔍 Ищем в Qdrant: {query}")
    results = await retriever.search(query, top_k=3)
    
    if not results:
        print("❌ Контекст не найден в Qdrant")
        return
    
    # Извлекаем текст из результатов
    contexts = [result["text"] for result in results]
    
    print(f"✅ Найдено {len(contexts)} контекстов")
    for i, ctx in enumerate(contexts[:2], 1):
        print(f"📄 Контекст {i}: {ctx[:1000]}...")
    
    # 4. Генерируем ответ
    print(f"🤖 Генерируем ответ...")
    response = await generation_service.generate(query, contexts)
    
    # 5. Результат
    print(f"\n❓ Вопрос: {query}")
    print(f"💡 Ответ: {response}")
    
    # 6. Закрываем соединения
    await retriever.close()
    
    print("\n✅ Тест пройден!")

if __name__ == "__main__":
    asyncio.run(test_generator_with_qdrant())