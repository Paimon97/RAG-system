import asyncio

class GenerationService:

    def __init__(self, llm_generator, prompt_builder, thread_pool):
        self.llm_generator = llm_generator
        self.prompt_builder = prompt_builder
        self.thread_pool = thread_pool

    async def generate(self, query: str, contexts: list[str]):

        prompt = self.prompt_builder.build_qa_prompt(query, contexts)
        # Получает текущий запущенный асинхронный цикл событий
        loop = asyncio.get_running_loop()
        # Запускает функцию generate в отдельном потоке из пула
        response = await loop.run_in_executor(
            self.thread_pool,
            self.llm_generator.generate,
            prompt
        )

        return response