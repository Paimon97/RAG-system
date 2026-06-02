# test_rag_system.py
import asyncio
import json
import time
import re
from typing import Dict, List
import aiohttp
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    test_id: str
    question: str
    expected_keywords: List[str]
    actual_answer: str
    confidence: float
    passed_keywords: bool
    passed_confidence: bool
    response_time: float
    error: str = None

class RAGSystemTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
    
    async def test_health(self) -> bool:
        """Проверка доступности системы"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"✅ Система здорова: {data}")
                        return True
                    else:
                        print(f"❌ Система недоступна: {resp.status}")
                        return False
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    async def query(self, question: str, top_k: int = 3) -> Dict:
        """Отправка запроса к RAG системе"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/query",
                json={"question": question, "top_k": top_k}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"HTTP {resp.status}", "answer": "", "confidence": 0}
    
    def check_keywords(self, answer: str, keywords: List[str]) -> bool:
        """Проверка наличия ключевых слов в ответе"""
        if not answer:
            return False
        answer_lower = answer.lower()
        for keyword in keywords:
            if keyword.lower() in answer_lower:
                return True
        return False
    
    async def run_test(self, test_case: Dict) -> TestResult:
        """Запуск одного теста"""
        start_time = time.time()
        
        try:
            response = await self.query(test_case["question"])
            response_time = time.time() - start_time
            
            actual_answer = response.get("answer", "")
            confidence = response.get("confidence", 0)
            
            passed_keywords = self.check_keywords(
                actual_answer, 
                test_case["expected_keywords"]
            )
            
            passed_confidence = confidence >= test_case.get("min_confidence", 0.3)
            
            return TestResult(
                test_id=test_case["id"],
                question=test_case["question"],
                expected_keywords=test_case["expected_keywords"],
                actual_answer=actual_answer,
                confidence=confidence,
                passed_keywords=passed_keywords,
                passed_confidence=passed_confidence,
                response_time=response_time
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case["id"],
                question=test_case["question"],
                expected_keywords=test_case["expected_keywords"],
                actual_answer="",
                confidence=0,
                passed_keywords=False,
                passed_confidence=False,
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    async def run_suite(self, test_suite_file: str = "test_suite.json"):
        """Запуск всего набора тестов"""
        # Проверяем существование файла
        import os
        if not os.path.exists(test_suite_file):
            print(f"⚠️ Файл {test_suite_file} не найден!")
            print("Создайте файл с тестами или укажите правильный путь")
            return
        
        with open(test_suite_file, 'r', encoding='utf-8') as f:
            suite = json.load(f)
        
        tests = suite["test_suite"]["tests"]
        total = len(tests)
        
        print(f"\n🧪 Запуск тестирования RAG системы")
        print(f"📋 Всего тестов: {total}")
        print("=" * 60)
        
        for i, test_case in enumerate(tests, 1):
            print(f"\n[{i}/{total}] Тест {test_case['id']}: {test_case['question'][:50]}...")
            result = await self.run_test(test_case)
            self.results.append(result)
            
            # Вывод результата
            status = "✅" if (result.passed_keywords or result.passed_confidence) else "❌"
            print(f"  {status} Уверенность: {result.confidence:.2f} | Время: {result.response_time:.2f}с")
            if result.actual_answer:
                # Очищаем ответ для отображения в консоли
                clean_answer = result.actual_answer.replace('\n', ' ').replace('\r', ' ')
                clean_answer = re.sub(r'\s+', ' ', clean_answer)
                print(f"  📝 Ответ: {clean_answer[:100]}...")
            if result.error:
                print(f"  ❌ Ошибка: {result.error}")
            
            # Небольшая задержка между запросами
            await asyncio.sleep(0.5)
        
        self.print_summary()
        self.print_readable_results()  # Автоматически сохраняем читаемый результат
    
    def print_summary(self):
        """Вывод сводки результатов"""
        total = len(self.results)
        if total == 0:
            print("Нет результатов для отображения")
            return
        
        passed_keywords = sum(1 for r in self.results if r.passed_keywords)
        passed_confidence = sum(1 for r in self.results if r.passed_confidence)
        
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("=" * 60)
        print(f"Всего тестов: {total}")
        print(f"✅ По ключевым словам: {passed_keywords}/{total} ({passed_keywords/total*100:.1f}%)")
        print(f"✅ По уверенности: {passed_confidence}/{total} ({passed_confidence/total*100:.1f}%)")
        
        # Средние показатели
        avg_confidence = sum(r.confidence for r in self.results) / total
        avg_response_time = sum(r.response_time for r in self.results) / total
        print(f"📈 Средняя уверенность: {avg_confidence:.2f}")
        print(f"⏱️  Среднее время ответа: {avg_response_time:.2f}с")
        
        # Проваленные тесты
        failed = [r for r in self.results if not (r.passed_keywords or r.passed_confidence)]
        if failed:
            print(f"\n❌ Проваленные тесты ({len(failed)}):")
            for r in failed:
                print(f"  - {r.test_id}: {r.question}")
        
        # Сохраняем результаты в JSON
        self.save_results()
    
    def save_results(self, filename: str = "test_results.json"):
        """Сохранение результатов в JSON"""
        if not self.results:
            print("Нет результатов для сохранения")
            return
            
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(self.results),
                "passed_keywords": sum(1 for r in self.results if r.passed_keywords),
                "passed_confidence": sum(1 for r in self.results if r.passed_confidence),
                "avg_confidence": sum(r.confidence for r in self.results) / len(self.results),
                "avg_response_time": sum(r.response_time for r in self.results) / len(self.results)
            },
            "details": [
                {
                    "test_id": r.test_id,
                    "question": r.question,
                    "actual_answer": r.actual_answer,
                    "confidence": r.confidence,
                    "passed_keywords": r.passed_keywords,
                    "passed_confidence": r.passed_confidence,
                    "response_time": r.response_time,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены в {filename}")
    
    def print_readable_results(self, filename: str = "readable_results.txt"):
        """Сохраняет результаты в читаемом формате без \n"""
        if not self.results:
            print("Нет результатов для сохранения в читаемом формате")
            return
            
        with open(filename, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write("=" * 80 + "\n")
            f.write(f"ОТЧЕТ О ТЕСТИРОВАНИИ RAG СИСТЕМЫ\n")
            f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Общая статистика
            total = len(self.results)
            passed_keywords = sum(1 for r in self.results if r.passed_keywords)
            passed_confidence = sum(1 for r in self.results if r.passed_confidence)
            avg_confidence = sum(r.confidence for r in self.results) / total
            avg_response_time = sum(r.response_time for r in self.results) / total
            
            f.write("СТАТИСТИКА:\n")
            f.write(f"  Всего тестов: {total}\n")
            f.write(f"  Успешно по ключевым словам: {passed_keywords}/{total} ({passed_keywords/total*100:.1f}%)\n")
            f.write(f"  Успешно по уверенности: {passed_confidence}/{total} ({passed_confidence/total*100:.1f}%)\n")
            f.write(f"  Средняя уверенность: {avg_confidence:.2f}\n")
            f.write(f"  Среднее время ответа: {avg_response_time:.2f}с\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Детальные результаты
            f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n\n")
            
            for i, r in enumerate(self.results, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"ТЕСТ #{i} - {r.test_id}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Вопрос: {r.question}\n")
                f.write(f"Уверенность: {r.confidence:.2f}\n")
                f.write(f"Время ответа: {r.response_time:.2f}с\n")
                f.write(f"Статус: {'✅ ПРОЙДЕН' if (r.passed_keywords or r.passed_confidence) else '❌ ПРОВАЛЕН'}\n")
                
                # Очищаем ответ от \n для читаемости
                clean_answer = r.actual_answer.replace('\n', ' ').replace('\r', ' ')
                clean_answer = re.sub(r'\s+', ' ', clean_answer)
                
                # # Ограничиваем длину ответа для читаемости (опционально)
                # if len(clean_answer) > 500:
                #     clean_answer = clean_answer[:500] + "..."
                
                f.write(f"\nОтвет:\n{clean_answer}\n")
                
                # Ожидаемые ключевые слова
                if r.expected_keywords:
                    f.write(f"\nОжидаемые ключевые слова: {', '.join(r.expected_keywords)}\n")
                    found_keywords = [kw for kw in r.expected_keywords if kw.lower() in r.actual_answer.lower()]
                    if found_keywords:
                        f.write(f"Найдено ключевых слов: {', '.join(found_keywords)}\n")
                    else:
                        f.write(f"Ключевые слова НЕ найдены!\n")
                
                if r.error:
                    f.write(f"\nОшибка: {r.error}\n")
                
                f.write(f"\n{'='*60}\n")
        
        print(f"\n📄 Читаемый отчет сохранен в {filename}")

async def main():
    tester = RAGSystemTester()
    
    # Проверяем здоровье системы
    if not await tester.test_health():
        print("\n❌ Система не запущена!")
        print("Запустите сервер командой: python -m src.main")
        print("Убедитесь, что сервер работает на http://127.0.0.1:8000")
        return
    
    # Запускаем тесты
    await tester.run_suite("test_suite.json")

if __name__ == "__main__":
    asyncio.run(main())