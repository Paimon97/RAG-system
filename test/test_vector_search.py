import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict
import numpy as np

from src.services.retriever.retriever import HybridRetriever
from src.config import settings


class TestHybridRetriever:
    """Тесты для HybridRetriever с фокусом на _vector_search"""
    
    @pytest.fixture
    def mock_embedder(self):
        """Создает мок для эмбеддера"""
        embedder = Mock()
        
        # Мокаем encode_batch, чтобы возвращал тестовые векторы
        def mock_encode_batch(texts, is_query=False):
            # Создаем случайный вектор размером VECTOR_SIZE
            vectors = np.random.rand(len(texts), settings.VECTOR_SIZE).astype(np.float32)
            # Нормализуем для косинусного расстояния
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors
        
        embedder.encode_batch = mock_encode_batch
        return embedder
    
    @pytest.fixture
    def mock_qdrant(self):
        """Создает мок для Qdrant клиента"""
        qdrant = AsyncMock()
        
        # Мокаем ответ query_points
        mock_response = Mock()
        mock_response.points = [
            Mock(
                payload={
                    "text": "Это тестовый документ номер 1",
                    "metadata": {"source": "test1.txt", "page": 1}
                },
                score=0.95
            ),
            Mock(
                payload={
                    "text": "Это тестовый документ номер 2",
                    "metadata": {"source": "test2.txt", "page": 2}
                },
                score=0.87
            ),
            Mock(
                payload={
                    "text": "Это тестовый документ номер 3",
                    "metadata": {"source": "test3.txt", "page": 3}
                },
                score=0.76
            )
        ]
        qdrant.query_points.return_value = mock_response
        return qdrant
    
    @pytest.fixture
    def retriever(self, mock_embedder, mock_qdrant):
        """Создает экземпляр HybridRetriever с моками"""
        retriever = HybridRetriever(embedder=mock_embedder)
        retriever.qdrant = mock_qdrant
        retriever.collection_name = "test_collection"
        return retriever
    
    @pytest.mark.asyncio
    async def test_vector_search_basic(self, retriever):
        """Тест 1: Базовая работа _vector_search"""
        # Выполняем поиск
        results = await retriever._vector_search("тестовый запрос", top_k=3)
        
        # Проверяем результат
        assert len(results) == 3
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        
        # Проверяем структуру каждого результата
        for result in results:
            assert "text" in result
            assert "metadata" in result
            assert "score" in result
            assert "source" in result
            assert result["source"] == "semantic"
            assert isinstance(result["score"], float)
    
    @pytest.mark.asyncio
    async def test_vector_search_top_k(self, retriever):
        """Тест 2: Проверка параметра top_k"""
        # Тестируем с разными top_k
        results_k1 = await retriever._vector_search("запрос", top_k=1)
        results_k5 = await retriever._vector_search("запрос", top_k=5)
        
        assert len(results_k1) == 1
        assert len(results_k5) == 3  # У нас только 3 мок-результата
        
        # Проверяем, что top_k=3 вернет максимум 3 результата
        assert len(results_k5) <= 5
    
    @pytest.mark.asyncio
    async def test_vector_search_empty_query(self, retriever):
        """Тест 3: Пустой запрос"""
        results = await retriever._vector_search("", top_k=3)
        # Должен вернуть пустой список или обработать gracefully
        assert isinstance(results, list)
        # В реальности может быть пустой список или случайные результаты
    
    @pytest.mark.asyncio
    async def test_vector_search_very_long_query(self, retriever):
        """Тест 4: Очень длинный запрос (1000+ символов)"""
        long_query = "тест " * 500  # 1000 символов
        results = await retriever._vector_search(long_query, top_k=3)
        assert isinstance(results, list)
        # Не должно быть исключений
    
    @pytest.mark.asyncio
    async def test_vector_search_no_results(self, retriever, mock_qdrant):
        """Тест 5: Поиск не находит результатов"""
        # Мокаем пустой ответ
        empty_response = Mock()
        empty_response.points = []
        mock_qdrant.query_points.return_value = empty_response
        
        results = await retriever._vector_search("запрос без результатов", top_k=3)
        
        assert results == []
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_vector_search_without_payload(self, retriever, mock_qdrant):
        """Тест 6: Точки без payload"""
        # Мокаем ответ с точками без payload
        response_without_payload = Mock()
        point_without_payload = Mock()
        point_without_payload.payload = None
        point_without_payload.score = 0.5
        
        point_with_payload = Mock()
        point_with_payload.payload = {"text": "есть текст", "metadata": {}}
        point_with_payload.score = 0.9
        
        response_without_payload.points = [point_without_payload, point_with_payload]
        mock_qdrant.query_points.return_value = response_without_payload
        
        results = await retriever._vector_search("запрос", top_k=3)
        
        # Должны получить только точку с payload
        assert len(results) == 1
        assert results[0]["text"] == "есть текст"
    
    @pytest.mark.asyncio
    async def test_vector_search_correct_query_format(self, retriever, mock_embedder):
        """Тест 7: Проверка формата запроса к эмбеддеру"""
        # Создаем шпиона для encode_batch
        mock_embedder.encode_batch = Mock(wraps=mock_embedder.encode_batch)
        
        await retriever._vector_search("тестовый запрос", top_k=3)
        
        # Проверяем, что encode_batch вызван с правильными параметрами
        mock_embedder.encode_batch.assert_called_once()
        args, kwargs = mock_embedder.encode_batch.call_args
        
        # Проверяем, что передан список с одним запросом
        assert len(args[0]) == 1
        assert args[0][0] == "тестовый запрос"
        
        # Проверяем, что is_query=True
        assert kwargs.get("is_query") == True
    
    @pytest.mark.asyncio
    async def test_vector_search_qdrant_call(self, retriever, mock_qdrant):
        """Тест 8: Проверка вызова Qdrant с правильными параметрами"""
        await retriever._vector_search("тестовый запрос", top_k=5)
        
        # Проверяем, что query_points вызван один раз
        mock_qdrant.query_points.assert_called_once()
        
        # Получаем параметры вызова
        call_args = mock_qdrant.query_points.call_args
        kwargs = call_args[1]
        
        # Проверяем параметры
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["limit"] == 5
        assert kwargs["with_payload"] == True
        assert "query" in kwargs
        assert isinstance(kwargs["query"], list)  # Должен быть список
    
    @pytest.mark.asyncio
    async def test_vector_search_score_values(self, retriever):
        """Тест 9: Проверка, что scores в разумных пределах"""
        results = await retriever._vector_search("запрос", top_k=3)
        
        for result in results:
            # Косинусное расстояние должно быть между -1 и 1, но для эмбеддингов обычно [0,1]
            assert -1 <= result["score"] <= 1
    
    @pytest.mark.asyncio
    async def test_vector_search_metadata_structure(self, retriever):
        """Тест 10: Проверка структуры метаданных"""
        results = await retriever._vector_search("запрос", top_k=3)
        
        for result in results:
            metadata = result["metadata"]
            assert isinstance(metadata, dict)
            # В моке мы добавили source и page
            if metadata:  # Если метаданные не пустые
                assert "source" in metadata or "page" in metadata
    
    @pytest.mark.asyncio
    async def test_vector_search_consistency(self, retriever):
        """Тест 11: Повторные вызовы возвращают одинаковую структуру"""
        results1 = await retriever._vector_search("запрос", top_k=3)
        results2 = await retriever._vector_search("запрос", top_k=3)
        
        # Структура должна быть одинаковой
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert set(r1.keys()) == set(r2.keys())
    
    @pytest.mark.asyncio
    async def test_vector_search_special_characters(self, retriever):
        """Тест 12: Запрос со спецсимволами"""
        special_queries = [
            "тест!@#$%^&*()",
            "вопрос? ответ.",
            "ключ-слово/слеш",
            "юникод: привет 🌍 мир",
            "цифры: 123 456 789",
        ]
        
        for query in special_queries:
            results = await retriever._vector_search(query, top_k=3)
            assert isinstance(results, list)
            # Не должно быть исключений


class TestVectorSearchIntegration:
    """Интеграционные тесты (требуют реальной БД)"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_qdrant_connection(self):
        """Тест с реальным подключением к Qdrant"""
        # Этот тест требует запущенного Qdrant
        from src.services.embedder import EmbeddingService
        from src.services.retriever.retriever import HybridRetriever
        
        embedder = EmbeddingService()
        retriever = HybridRetriever(embedder=embedder)
        
        try:
            await retriever.initialize()
            results = await retriever._vector_search("тестовый запрос", top_k=3)
            assert isinstance(results, list)
        finally:
            await retriever.close()
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_vector_search_performance(self, retriever):
        """Тест производительности"""
        import time
        
        start = time.time()
        for _ in range(10):
            await retriever._vector_search("тестовый запрос", top_k=5)
        elapsed = time.time() - start
        
        # 10 запросов должны выполняться меньше 2 секунд (для моков)
        assert elapsed < 2.0
        print(f"10 запросов выполнено за {elapsed:.2f} секунд")


# Дополнительный тест для отладки конкретной проблемы
class TestVectorSearchDebug:
    """Специальные тесты для отладки"""
    
    @pytest.mark.asyncio
    async def test_vector_format(self):
        """Проверка формата вектора - это то, что вызвало ошибку Conversion between multi and regular vectors"""
        from src.services.embedder import EmbeddingService
        
        embedder = EmbeddingService()
        
        # Получаем вектор
        query_vector = embedder.encode_batch(["тест"], is_query=True)
        
        # Проверяем формат
        print(f"Type: {type(query_vector)}")
        print(f"Shape: {query_vector.shape if hasattr(query_vector, 'shape') else 'N/A'}")
        print(f"NDim: {query_vector.ndim if hasattr(query_vector, 'ndim') else 'N/A'}")
        
        # Это должно быть 2D массив: (1, vector_size)
        assert query_vector.ndim == 2
        assert query_vector.shape[0] == 1
        assert query_vector.shape[1] == settings.VECTOR_SIZE
        
        # Проверяем, что query_vector[0] - плоский список
        vector_for_qdrant = query_vector[0].tolist()
        assert isinstance(vector_for_qdrant, list)
        # Проверяем, что это не список списков
        assert not isinstance(vector_for_qdrant[0], list)
        assert len(vector_for_qdrant) == settings.VECTOR_SIZE


# Запуск тестов:
# pytest tests/test_retriever.py -v
# Для конкретного теста:
# pytest tests/test_retriever.py::TestHybridRetriever::test_vector_search_basic -v