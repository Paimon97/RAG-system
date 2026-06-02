# test_generator_only.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.generator.llm_generator import SafeLLMGenerator

def test_generator():
    """Тестирование генератора на разных вопросах"""
    
    # Инициализируем генератор
    print("🔧 Инициализация генератора...")
    generator = SafeLLMGenerator()
    print("✅ Генератор готов\n")
    
    # Тестовые данные (контексты из вашего data1.txt)
    test_contexts = {
        "лаунчер": [
            "Для начала, Вам потребуется перейти на официальный веб-сайт проекта и скачать лаунчер. После установки лаунчера, Вы сможете загрузить игру, нажав на кнопку Загрузка внутри лаунчера."
        ],
        "промокод": [
            "Создание уникального промокода стоит 10.000.000 рублей. Вы можете использовать промокод любого ютубера, обычно о нём сообщают в роликах или в описании."
        ],
        "никнейм": [
            "Вам необходимо придумать уникальное имя в формате Имя_Фамилия на латинском алфавите. Примеры: Vyacheslav_Ivankov, Kirill_Litvin, Vladlena_Ivanova."
        ],
        "регистрация": [
            "При регистрации Вам необходимо придумать надежный пароль и ввести Вашу электронную почту. После этого нажмите на кнопку Зарегистрироваться."
        ],
        "скин": [
            "Вы можете выбрать скин, отражающий Вашу индивидуальность. В магазине одежды предлагается широкий выбор различных скинов, которые Вы можете приобрести за игровую валюту."
        ]
    }
    
    # Тестовые вопросы
    test_cases = [
        {
            "name": "Тест 1: Где скачать?",
            "query": "Где можно скачать лаунчер?",
            "contexts": test_contexts["лаунчер"]
        },
        {
            "name": "Тест 2: Сколько стоит?",
            "query": "Сколько стоит создание промокода?",
            "contexts": test_contexts["промокод"]
        },
        {
            "name": "Тест 3: Формат ника",
            "query": "В каком формате нужно создавать никнейм?",
            "contexts": test_contexts["никнейм"]
        },
        {
            "name": "Тест 4: Примеры ников",
            "query": "Приведи пример правильного ника",
            "contexts": test_contexts["никнейм"]
        },
        {
            "name": "Тест 5: Регистрация",
            "query": "Что нужно ввести при регистрации?",
            "contexts": test_contexts["регистрация"]
        },
        {
            "name": "Тест 6: Скины",
            "query": "Где можно купить скины?",
            "contexts": test_contexts["скин"]
        },
        {
            "name": "Тест 7: Пустой контекст",
            "query": "Что делать при ошибке?",
            "contexts": []  # Пустой контекст
        },
        {
            "name": "Тест 8: Несколько контекстов",
            "query": "Что такое промокод?",
            "contexts": test_contexts["промокод"] + test_contexts["лаунчер"]
        }
    ]
    
    print("=" * 70)
    print("🧪 ЗАПУСК ТЕСТИРОВАНИЯ GENERATOR")
    print("=" * 70)
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 {test['name']}")
        print(f"   Вопрос: {test['query']}")
        print(f"   Контекстов: {len(test['contexts'])}")
        
        # Генерируем ответ
        result = generator.generate(test['query'], test['contexts'])
        
        print(f"   📝 Ответ: {result['answer'][:150]}...")
        print(f"   📊 Уверенность: {result['confidence']:.2f}")
        
        # Оценка качества
        quality = "✅" if result['confidence'] > 0.5 else "⚠️"
        print(f"   {quality} Качество: {'Хорошо' if result['confidence'] > 0.5 else 'Низкое'}")
        
        results.append({
            "name": test['name'],
            "query": test['query'],
            "answer": result['answer'],
            "confidence": result['confidence']
        })
        
        print("-" * 50)
    
    # Статистика
    print("\n" + "=" * 70)
    print("📊 СТАТИСТИКА")
    print("=" * 70)
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    high_quality = sum(1 for r in results if r['confidence'] > 0.5)
    
    print(f"Всего тестов: {len(results)}")
    print(f"Средняя уверенность: {avg_confidence:.2f}")
    print(f"Хороших ответов (>0.5): {high_quality}/{len(results)}")
    
    return results

def test_edge_cases():
    """Тестирование граничных случаев"""
    print("\n" + "=" * 70)
    print("🔍 ТЕСТИРОВАНИЕ ГРАНИЧНЫХ СЛУЧАЕВ")
    print("=" * 70)
    
    generator = SafeLLMGenerator()
    
    edge_tests = [
        {
            "name": "Очень длинный вопрос",
            "query": "Где " * 100 + "скачать лаунчер?",
            "contexts": ["Скачать лаунчер можно на официальном сайте."]
        },
        {
            "name": "Очень длинный контекст",
            "query": "Где скачать лаунчер?",
            "contexts": ["Длинный текст. " * 200]
        },
        {
            "name": "Спецсимволы в запросе",
            "query": "Где скачать лаунчер?!!! @#$%",
            "contexts": ["Скачать лаунчер можно на сайте."]
        },
        {
            "name": "Пустой ответ генератора",
            "query": "Сложный вопрос без ответа",
            "contexts": ["Просто случайный текст."]
        }
    ]
    
    for test in edge_tests:
        print(f"\n📋 {test['name']}")
        result = generator.generate(test['query'], test['contexts'])
        print(f"   Ответ: {result['answer'][:100]}...")
        print(f"   Уверенность: {result['confidence']:.2f}")
        print(f"   Длина ответа: {len(result['answer'])} символов")

def test_accuracy_with_expected():
    """Тестирование точности с ожидаемыми ключевыми словами"""
    print("\n" + "=" * 70)
    print("🎯 ТЕСТИРОВАНИЕ ТОЧНОСТИ")
    print("=" * 70)
    
    generator = SafeLLMGenerator()
    
    accuracy_tests = [
        {
            "query": "Где скачать лаунчер?",
            "context": "Скачать лаунчер можно на официальном сайте проекта.",
            "expected": ["официальный", "сайт", "скачать"]
        },
        {
            "query": "Сколько стоит промокод?",
            "context": "Создание уникального промокода стоит 10.000.000 рублей.",
            "expected": ["10.000.000", "рублей"]
        },
        {
            "query": "Формат никнейма?",
            "context": "Никнейм должен быть в формате Имя_Фамилия.",
            "expected": ["Имя_Фамилия", "формат"]
        }
    ]
    
    passed = 0
    for test in accuracy_tests:
        result = generator.generate(test['query'], [test['context']])
        answer_lower = result['answer'].lower()
        
        found = [kw for kw in test['expected'] if kw.lower() in answer_lower]
        is_passed = len(found) > 0
        
        print(f"\nВопрос: {test['query']}")
        print(f"Ответ: {result['answer'][:100]}")
        print(f"Ожидаемые слова: {', '.join(test['expected'])}")
        print(f"Найдено: {', '.join(found) if found else 'НИЧЕГО'}")
        print(f"Результат: {'✅' if is_passed else '❌'}")
        
        if is_passed:
            passed += 1
    
    print(f"\nТочность: {passed}/{len(accuracy_tests)} ({passed/len(accuracy_tests)*100:.0f}%)")

if __name__ == "__main__":
    # Запускаем все тесты
    test_generator()
    # test_edge_cases()
    # test_accuracy_with_expected()