from src.models.schemas import QueryResponse
import asyncio

class RAGOrchestrator:

    def __init__(self, retrieval, generation_service, validator, cache):
        self.retrieval = retrieval
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
                 self.retrieval.search(question, top_k),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            return QueryResponse(answer="Поиск занял слишком много времени", confidence=0)
        
        if not contexts:
            return QueryResponse(
                answer="Информация не найдена",
                confidence=0
            )

        # Вытаскиваем текст из metadata
        context_texts = [
            c["text"]
            for c in contexts
        ]

        # GENERATION
        try:
            generated = await asyncio.wait_for(
            self.generation.generate(question, context_texts),
            timeout=10.0
            )
        except asyncio.TimeoutError:
            return QueryResponse(
            answer=context_texts[0],
            confidence=0.5
            )
    # try:
    #     response = await asyncio.wait_for(
    #         asyncio.get_event_loop().run_in_executor(
    #             thread_pool,
    #             generator.generate,
    #             request.question,
    #             [doc["text"] for doc in context]
    #         ),
    #         timeout=10.0
    #     )
    # except asyncio.TimeoutError:
    #     return QueryResponse(
    #         answer=context[0]["text"],
    #         confidence=0.5,
    #         sources=[doc["metadata"] for doc in context]
    #     )
    
        # VALIDATION
        is_valid = self.validator.is_hallucination(
            generated["answer"],
            context_texts[0]
        )

        if is_valid:

            return QueryResponse(
                answer="Ответ не прошел проверку",
                confidence=0
            )

        result = QueryResponse(
            answer=generated["answer"],
            confidence=generated["confidence"],
            sources=[
                c["metadata"]
                for c in contexts
            ]
        )

        # SAVE CACHE
        self.cache.set(
            question,
            result
        )

        return result