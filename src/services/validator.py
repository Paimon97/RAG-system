import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class HallucinationGuard:
    def __init__(self, embedder):
        self.nlp = spacy.load("ru_core_news_sm")
        self.similarity_model = embedder
    
    def is_hallucination(self, answer: str, context: str) -> bool:
        # Извлечение сущностей
        answer_entities = set([ent.text.lower() for ent in self.nlp(answer).ents])
        context_entities = set([ent.text.lower() for ent in self.nlp(context).ents])
        
        # Новые сущности = галлюцинация
        if answer_entities - context_entities:
            return True
        
        # Семантическая близость
        answer_emb = self.similarity_model.encode([answer])
        context_emb = self.similarity_model.encode([context])
        similarity = cosine_similarity(answer_emb, context_emb)[0][0]
        
        return similarity < 0.6
    

#     # validator.py - исправленная версия
# import spacy
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# class HallucinationGuard:
#     def __init__(self, embedder):  # ← Принимаем embedder из container.py
#         self.nlp = spacy.load("ru_core_news_sm")
#         self.embedder = embedder  # ← Используем существующую модель
    
#     def is_hallucination(self, answer: str, context: str) -> bool:
#         # Извлечение сущностей
#         answer_entities = set([ent.text.lower() for ent in self.nlp(answer).ents])
#         context_entities = set([ent.text.lower() for ent in self.nlp(context).ents])
        
#         # Новые сущности = галлюцинация
#         if answer_entities - context_entities:
#             return True
        
#         # Семантическая близость через существующий embedder
#         answer_emb = self.embedder.encode_query(answer)  # Переиспользуем!
#         context_emb = self.embedder.encode_passage(context)
        
#         # Нормализуем и считаем косинусное расстояние
#         similarity = np.dot(answer_emb, context_emb) / (np.linalg.norm(answer_emb) * np.linalg.norm(context_emb))
        
#         return similarity < 0.6

# # Было:
# validator = HallucinationGuard()

# # Стало:
# validator = HallucinationGuard(embedder)  # Передаем embedder