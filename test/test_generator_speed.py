# test_generator_speed.py
import sys
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from services.generator.llm_generator import SafeLLMGenerator

def test_speed():
    print("=" * 60)
    print("🚀 ТЕСТ СКОРОСТИ ГЕНЕРАТОРА")
    print("=" * 60)
    
    generator = SafeLLMGenerator()
    
    # Тестовые данные
    contexts = [
        "Лаунчер можно скачать на официальном сайте проекта по адресу example.com. Установка занимает около 5 минут."
    ]
    
    queries = [
        "Где скачать лаунчер?",
        "Сколько времени занимает установка?",
        "Какой адрес сайта?",
    ]
    
    times = []
    
    for query in queries:
        start = time.time()
        result = generator.generate(query, contexts)
        elapsed = time.time() - start
        
        times.append(elapsed)
        
        print(f"\nВопрос: {query}")
        print(f"Ответ: {result['answer'][:100]}")
        print(f"Время: {elapsed:.3f} сек")
        print(f"Уверенность: {result['confidence']:.2f}")
    
    print("\n" + "=" * 60)
    print(f"📊 СТАТИСТИКА")
    print(f"Среднее время: {sum(times)/len(times):.3f} сек")
    print(f"Минимальное время: {min(times):.3f} сек")
    print(f"Максимальное время: {max(times):.3f} сек")
    
    if hasattr(generator, 'get_stats'):
        print(f"\nКэш статистика: {generator.get_stats()}")

if __name__ == "__main__":
    test_speed()