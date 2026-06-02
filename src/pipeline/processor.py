from src.config import settings
from transformers import pipeline

import asyncio
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, nlp_model, chunker, embedder, retriever):
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.nlp = nlp_model
    
    async def process_document(self, file_path: str):
        # Чтение файла
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = self.chunker._spacy_split(text)

        # Ограничиваем параллелизм
        semaphore = asyncio.Semaphore(10)  # Не более 10 чанков одновременно
        
        async def process_with_limit(chunk, i):
            async with semaphore:
                return await self._process_chunk(chunk, i, file_path, len(chunks))
        
        tasks = [process_with_limit(chunk, i) for i, chunk in enumerate(chunks)]
        metadata = await asyncio.gather(*tasks)
        
        await self.retriever.add_document(chunks, metadata)
        return metadata

    """Обработка одного чанка"""
    async def _process_chunk(self, chunk: str, idx: int, file_path: str, total_chunks: int):

        # Только теги, если нужны
        loop = asyncio.get_event_loop()
        tags = await loop.run_in_executor(None, self._extract_tags, chunk)
            
        # # Генерируем эмбеддинг для чанка
        # embedding = await loop.run_in_executor(
        #     None, 
        #     self.embedder.encode_passage, 
        #     chunk
        # )
            
        return {
            "source": file_path,
            "chunk_index": idx,
            "tags": tags,
            "chunk_length": len(chunk),
            "total_chunks": total_chunks
        }


    def _extract_tags(self, chunk: str) -> list:
        """Синхронное извлечение тегов"""
        try:
            doc = self.nlp(chunk[:1000])
            tags = list(set([token.lemma_ for token in doc if token.pos_ == "NOUN"]))
            return tags[:10]
        except Exception as e:
            logger.error(f"Tag extraction error: {e}")
            return []