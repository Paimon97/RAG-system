import asyncio

class GenerationService:

    def __init__(self, generator, prompt_builder, thread_pool):
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.thread_pool = thread_pool

    async def generate(self, query: str, contexts: list[str]):

        prompt = self.prompt_builder.build_qa_prompt(query, contexts)

        loop = asyncio.get_running_loop()

        response = await loop.run_in_executor(
            self.thread_pool,
            self.generator.generate,
            prompt
        )

        return response