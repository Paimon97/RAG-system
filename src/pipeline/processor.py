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
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()

        sections = self.section_parser.extract_sections(html)

        chunks = []
        base_metadata = []

        for section in sections:

            section_chunks = self.chunker._spacy_split(
                section["text"]
            )

            for chunk in section_chunks:

                enriched_chunk = "\n".join(
                    filter(
                        None,
                        [
                            section["h1"],
                            section["h2"],
                            section["h3"],
                            chunk
                        ]
                    )
                )

                chunks.append(enriched_chunk)

                base_metadata.append({
                    "source": file_path,
                    "h1": section["h1"],
                    "h2": section["h2"],
                    "h3": section["h3"]
                })

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

    async def process_sections(self, sections: list[dict], source_url: str):

        chunks = []
        metadata = []

        for section in sections:

            section_chunks = self.chunker._spacy_split(section["text"])

            for chunk in section_chunks:
                enriched_chunk = "\n".join(
                    filter(
                        None,
                        [
                            section["h1"],
                            section["h2"],
                            section["h3"],
                            chunk
                        ]
                    )
                )

                chunks.append(enriched_chunk)

                metadata.append(
                    {
                        "source_url": source_url,
                        "h1": section["h1"],
                        "h2": section["h2"],
                        "h3": section["h3"]
                    }
                )

        semaphore = asyncio.Semaphore(10)

        async def process_with_limit(chunk, idx, base_meta):
            async with semaphore:
                return await self._process_chunk_with_metadata(chunk, idx, base_meta, len(chunks))

        tasks = [
            process_with_limit(
                chunk,
                i,
                metadata[i]
            )

            for i, chunk in enumerate(chunks)
        ]

        final_metadata = await asyncio.gather(*tasks)

        await self.document_manager.add_document(chunks, final_metadata)

        return final_metadata
    
    async def _process_chunk_with_metadata(self, chunk: str, idx: int, base_metadata: dict, total_chunks: int):

        loop = asyncio.get_event_loop()

        tags = await loop.run_in_executor(None, self._extract_tags, chunk)

        return {
            **base_metadata,
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