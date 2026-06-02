from src.config import settings


class TextChunker:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def _spacy_split(self, text: str) -> list[str]:
        """
        Простая разбивка текста на абзацы.
        Каждый абзац становится отдельным чанком.
        """
        # Разбиваем на абзацы
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Если нет двойных переносов, пробуем одиночные
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Если все еще нет абзацев, возвращаем весь текст
        if not paragraphs:
            return [text.strip()] if text.strip() else []
        
        # Фильтруем слишком большие абзацы (опционально)
        max_tokens = 1024
        filtered_paragraphs = []
        
        for para in paragraphs:
            para_tokens = len(para.split())
            if para_tokens <= max_tokens:
                filtered_paragraphs.append(para)
            else:
                # Если абзац слишком большой, разбиваем его на предложения
                doc = self.nlp(para)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                
                current_chunk = []
                current_tokens = 0
                
                for sentence in sentences:
                    sent_tokens = len(sentence.split())
                    if current_tokens + sent_tokens > max_tokens:
                        if current_chunk:
                            filtered_paragraphs.append(" ".join(current_chunk))
                            current_chunk = []
                            current_tokens = 0
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
                
                if current_chunk:
                    filtered_paragraphs.append(" ".join(current_chunk))
        
        return filtered_paragraphs if filtered_paragraphs else [text]



    # Разбивка с использованием spaCy (лучше качество)
    # def _spacy_split(self, text: str) -> list[str]:
    #     doc = self.nlp(text)

    #     max_tokens = settings.CHUNK_SIZE
    #     overlap = settings.CHUNK_OVERLAP

    #     # Разбиваем текст на предложения
    #     sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    #     chunks = []

    #     current_chunk = []
    #     current_tokens = 0

    #     for sentence in sentences:

    #         # Быстрый подсчет токенов
    #         sent_tokens = len(sentence.split())

    #         # Если предложение слишком большое
    #         if sent_tokens > max_tokens:
    #             continue

    #         # Если превышаем лимит → сохраняем чанк
    #         if current_tokens + sent_tokens > max_tokens:

    #             chunk_text = " ".join(current_chunk)
    #             chunks.append(chunk_text)

    #             # ---------- OVERLAP ----------
    #             overlap_chunk = []
    #             overlap_tokens = 0

    #             # Идем с конца текущего чанка
    #             for prev_sentence in reversed(current_chunk):

    #                 prev_tokens = len(prev_sentence.split())

    #                 if overlap_tokens + prev_tokens > overlap:
    #                     break

    #                 overlap_chunk.insert(0, prev_sentence)
    #                 overlap_tokens += prev_tokens

    #             # Новый чанк начинается с overlap
    #             current_chunk = overlap_chunk
    #             current_tokens = overlap_tokens

    #         # Добавляем текущее предложение
    #         current_chunk.append(sentence)
    #         current_tokens += sent_tokens

    #     # Последний чанк
    #     if current_chunk:
    #         chunks.append(" ".join(current_chunk))

    #     return chunks


