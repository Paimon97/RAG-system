from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from src.config import settings
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device="cpu"
        )
        self.model.eval()
        
        # Установка максимальной длины для токенизатора
        self.max_seq_length = 512
        self.model.max_seq_length = self.max_seq_length
        
        # Явно задаем max_length для токенизатора
        if hasattr(self.model, 'tokenizer'):
            self.model.tokenizer.model_max_length = self.max_seq_length
    
    @torch.no_grad()
    def encode_query(self, text: str) -> np.ndarray:
    
        return self.model.encode(
            f"query: {text}",
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    @torch.no_grad()
    def encode_passage(self, text: str) -> np.ndarray:
        return self.model.encode(
            f"passage: {text}",
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    @torch.no_grad()
    def encode_batch(self, texts: list, is_query: bool = False) -> np.ndarray:
        prefix = "query:" if is_query else "passage: "
        texts = [prefix + t for t in texts]
        
        # Кодируем с автоматическим обрезанием
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=8  # Маленький батч для CPU
        )