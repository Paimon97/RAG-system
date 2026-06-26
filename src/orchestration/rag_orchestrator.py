from src.models.schemas import QueryResponse
import asyncio
import logging

logger = logging.getLogger(__name__)

class RAGOrchestrator:

    def __init__(self, retriever, generation_service, validator, cache):
        self.retriever = retriever
        self.generation = generation_service
        self.validator = validator
        self.cache = cache

    async def answer(self, question: str, top_k: int) -> QueryResponse:
        # CACHE
        # cached = self.cache.get(question)
        # if cached:
        #     return cached

        # RETRIEVAL
        try:
            contexts = await asyncio.wait_for(
                self.retriever.search(question, top_k),
                timeout=50.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Search timeout for: {question[:50]}")
            return QueryResponse(answer="Поиск занял слишком много времени", confidence=0)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return QueryResponse(answer=f"Ошибка поиска: {str(e)}", confidence=0)
        
        # Проверяем контексты
        if contexts is None or len(contexts) == 0:
            logger.warning(f"No contexts found for: {question[:50]}")
            return QueryResponse(
                answer="Информация не найдена",
                confidence=0
            )

        # Вытаскиваем текст из результатов поиска
        context_texts = []
        for c in contexts:
            if c and isinstance(c, dict) and "text" in c:
                context_texts.append(c["text"])
        
        if not context_texts:
            return QueryResponse(
                answer="Не удалось извлечь текст из найденных документов",
                confidence=0
            )

        try:
            answer_text = await asyncio.wait_for(
                self.generation.generate(question, context_texts),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Generation timeout for: {question[:100]}")
            # Используем первый контекст как fallback
            fallback_answer = context_texts[0][:1000] if context_texts else "Время генерации истекло"
            return QueryResponse(
                answer=fallback_answer,
                confidence=0.5,
                sources=[c.get("metadata", {}) for c in contexts if c],
                context=" ".join(context_texts)
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return QueryResponse(
                answer=f"Ошибка генерации: {str(e)}",
                confidence=0
            )
        
        try:
            # Объединяем контексты для проверки
            combined_context = " ".join(context_texts[:2]) if context_texts else ""
            
            is_hallucination = self.validator.is_hallucination(
                answer_text,  # ← теперь передаем строку
                combined_context
            )
            
            if is_hallucination:
                logger.warning(f"Hallucination detected for: {question[:100]}")
                return QueryResponse(
                    answer="Ответ не прошел проверку на достоверность",
                    confidence=0,
                    sources=[c.get("metadata", {}) for c in contexts if c]
                )
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # Если валидация упала, все равно возвращаем ответ
        
        # Вычисляем уверенность на основе количества найденных контекстов
        confidence = min(0.9, len(contexts) * 0.3)  # Чем больше контекстов, тем выше уверенность
        
        context_str = "\n\n".join([c.get("text", "") for c in contexts if c and isinstance(c, dict)])

        result = QueryResponse(
            answer=answer_text, 
            confidence=confidence,
            sources=[c.get("metadata", {}) for c in contexts if c],
            context=context_str
        )

        # # SAVE CACHE
        # try:
        #     self.cache.set(question, result)
        # except Exception as e:
        #     logger.error(f"Cache save error: {e}")

        return result



        # results = await self.retriever.search(question, top_k=3)
        # contexts = [result["text"] for result in results]
        # answer_text = await self.generation.generate(question, contexts)
        # await self.retriever.close()