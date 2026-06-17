# Код реализует асинхронный конвейер для обработки текстовых документов. 
# Он разбивает большой текст на части, извлекает из них ключевые слова (теги) и сохраняет результат.

from src.config import settings
import asyncio
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, nlp_model, chunker, embedder, document_manager):
        self.chunker = chunker
        self.embedder = embedder
        self.document_manager = document_manager  
        self.nlp = nlp_model
    
    async def process_document(self, file_path: str):
        # Чтение файла
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        """Перед разбитием на чанки надо добавить подготовку текста"""

        # Разбитие на чанки
        chunks = self.chunker._spacy_split(text)
        
        # Семафор для ограничения количества одновременно выполняемых задач 
        semaphore = asyncio.Semaphore(10)
        
        # Функция обертка которая гарантирует, что тело задачи будет выполненно только после захвата семафора
        async def process_with_limit(chunk, i):
            async with semaphore:
                return await self._process_chunk(chunk, i, file_path, len(chunks))
        
        # Все задачи запускаются одновременно с помощью gather
        tasks = [process_with_limit(chunk, i) for i, chunk in enumerate(chunks)]
        metadata = await asyncio.gather(*tasks)
        
        # Используем document_manager для добавления
        await self.document_manager.add_document(chunks, metadata)
        return metadata

    # Этот метод обрабатывает один фрагмент текста
    async def _process_chunk(self, chunk: str, idx: int, file_path: str, total_chunks: int):
        loop = asyncio.get_event_loop() # Получение событийного цикла
        # run_in_executor отправляет вызов функции в пул потоков. Тем самым не блокируя основной поток
        tags = await loop.run_in_executor(None, self._extract_tags, chunk) 
            
        return {
            "source": file_path,
            "chunk_index": idx,
            "tags": tags,
            "chunk_length": len(chunk),
            "total_chunks": total_chunks
        }

    """КАК СЕЙЧАС ИСПОЛЬЗУЮТСЯ ТЕГИ?"""
    # Метод для извлечения тегов
    def _extract_tags(self, chunk: str) -> list:
        try:
            # Объект nlp модели 
            doc = self.nlp(chunk[:1000])
            # Фильтруются токены, являющиеся существительными.
            # Для каждого существительного берется его лемма (базовая словарная форма).
            tags = list(set([token.lemma_ for token in doc if token.pos_ == "NOUN"]))
            return tags[:10]
        except Exception as e:
            logger.error(f"Tag extraction error: {e}")
            return []