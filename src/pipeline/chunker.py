# Разбивает текст на предложения и собирает предложения в чанки, пока не превышен лимит токенов.

from collections import deque
from src.config import settings

class TextChunker:
    def __init__(self, nlp_model, tokenizer):
        self.nlp = nlp_model
        self.tokenizer = tokenizer

    def _count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(text, add_special_tokens=False)
        )

    def _split_long_sentence(self, sentence: str, max_tokens: int) -> list[str]:
        """
        Если предложение превышает лимит токенов,
        режем его по токенам токенизатора.
        """
        token_ids = self.tokenizer.encode(sentence, add_special_tokens=False)

        chunks = []

        for i in range(0, len(token_ids), max_tokens):
            piece = token_ids[i:i + max_tokens]

            chunks.append(self.tokenizer.decode(piece, skip_special_tokens=True))

        return chunks

    # Принимает строку и возвращает список чанков
    def _spacy_split(self, text: str) -> list[str]:
        doc = self.nlp(text)

        max_tokens = settings.CHUNK_SIZE
        min_chunk_tokens = settings.MIN_CHUNK_SIZE
        overlap_tokens_limit = settings.CHUNK_OVERLAP

        # Разбиваем текст на предложения. 
        # doc.sents — генератор, который возвращает предложения
        # Фильтруем пустые предложения с помощью if sent.text.strip(). 
        # Удаляем лишние пробелы в начале и в конце.
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        processed_sentences = []

        # Обработка длинных предложений
        for sentence in sentences:

            sent_tokens = self._count_tokens(sentence)

            if sent_tokens <= max_tokens:
                processed_sentences.append((sentence, sent_tokens))
                continue

            subchunks = self._split_long_sentence(sentence, max_tokens)

            for subchunk in subchunks:
                processed_sentences.append(
                    (
                        subchunk,
                        self._count_tokens(subchunk)
                    )
                )

        chunks = []

        current_sentences = []
        current_tokens = 0

        for sentence, sent_tokens in processed_sentences:

            if current_tokens + sent_tokens > max_tokens:

                chunk_text = " ".join(current_sentences)

                if chunk_text.strip():
                    chunks.append(chunk_text)

                # OVERLAP
                overlap_sentences = deque()
                overlap_tokens = 0

                for prev_sentence in reversed(current_sentences):

                    prev_tokens = self._count_tokens(prev_sentence)

                    if (overlap_tokens + prev_tokens > overlap_tokens_limit):
                        break

                    overlap_sentences.appendleft(prev_sentence)

                    overlap_tokens += prev_tokens

                current_sentences = list(overlap_sentences)

                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        # Удаляем слишком маленькие чанки
        filtered_chunks = []

        for chunk in chunks:

            chunk_tokens = self._count_tokens(chunk)

            if (filtered_chunks and chunk_tokens < min_chunk_tokens):
                filtered_chunks[-1] += " " + chunk
            else:
                filtered_chunks.append(chunk)

        return filtered_chunks