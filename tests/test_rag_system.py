# test_rag_system.py
import asyncio
import json
import time
import re
import os
import sys
from typing import Dict, List, Optional
from pathlib import Path
import aiohttp
from dataclasses import dataclass, asdict
from datetime import datetime

# Добавляем корень проекта в PATH для импорта
sys.path.insert(0, str(Path(__file__).parent))

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
    sources: List[Dict] = None
    error: str = None

class RAGSystemTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health(self) -> bool:
        """Проверка доступности системы"""
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Система здорова: {data}")
                    return True
                else:
                    print(f"❌ Система недоступна: {resp.status}")
                    return False
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            print(f"   Убедитесь, что сервер запущен: uvicorn src.main:app --reload")
            return False
    
    async def query(self, question: str, top_k: int = 3) -> Dict:
        """Отправка запроса к RAG системе"""
        try:
            async with self.session.post(
                f"{self.base_url}/query",
                json={"question": question, "top_k": top_k}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    return {
                        "error": f"HTTP {resp.status}: {error_text}",
                        "answer": "",
                        "confidence": 0,
                        "sources": []
                    }
        except Exception as e:
            return {
                "error": str(e),
                "answer": "",
                "confidence": 0,
                "sources": []
            }
    
    def check_keywords(self, answer: str, keywords: List[str]) -> bool:
        """Проверка наличия хотя бы одного ключевого слова в ответе"""
        if not answer:
            return False
        answer_lower = answer.lower()
        for keyword in keywords:
            if keyword.lower() in answer_lower:
                return True
        return False
    
    def check_all_keywords(self, answer: str, keywords: List[str]) -> Dict:
        """Проверка всех ключевых слов (возвращает статистику)"""
        if not answer:
            return {"found": [], "missing": keywords, "all_found": False}
        
        answer_lower = answer.lower()
        found = [kw for kw in keywords if kw.lower() in answer_lower]
        missing = [kw for kw in keywords if kw.lower() not in answer_lower]
        
        return {
            "found": found,
            "missing": missing,
            "all_found": len(missing) == 0
        }
    
    async def run_test(self, test_case: Dict) -> TestResult:
        """Запуск одного теста"""
        start_time = time.time()
        
        try:
            response = await self.query(
                test_case["question"],
                test_case.get("top_k", 3)
            )
            response_time = time.time() - start_time
            
            actual_answer = response.get("answer", "")
            confidence = response.get("confidence", 0)
            sources = response.get("sources", [])
            
            # Проверка ключевых слов (хотя бы одно)
            passed_keywords = self.check_keywords(
                actual_answer, 
                test_case["expected_keywords"]
            )
            
            # Проверка уверенности
            min_confidence = test_case.get("min_confidence", 0.3)
            passed_confidence = confidence >= min_confidence
            
            # Дополнительная проверка: есть ли источники
            has_sources = len(sources) > 0 if test_case.get("expect_sources", True) else True
            
            return TestResult(
                test_id=test_case["id"],
                question=test_case["question"],
                expected_keywords=test_case["expected_keywords"],
                actual_answer=actual_answer,
                confidence=confidence,
                passed_keywords=passed_keywords,
                passed_confidence=passed_confidence and has_sources,
                response_time=response_time,
                sources=sources,
                error=None
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
                sources=[],
                error=str(e)
            )
    
    async def run_suite(self, test_suite_file: str = "test_suite.json"):
        """Запуск всего набора тестов"""
        # Проверяем существование файла
        if not os.path.exists(test_suite_file):
            print(f"⚠️ Файл {test_suite_file} не найден!")
            print("Создаю пример тестового набора...")
            self.create_example_test_suite(test_suite_file)
            print(f"✅ Создан пример файла: {test_suite_file}")
            print("Отредактируйте его под свои вопросы и запустите снова.")
            return
        
        with open(test_suite_file, 'r', encoding='utf-8') as f:
            suite = json.load(f)
        
        tests = suite.get("test_suite", {}).get("tests", [])
        if not tests:
            print("❌ В файле нет тестов!")
            return
        
        total = len(tests)
        
        print(f"\n🧪 Запуск тестирования RAG системы")
        print(f"📋 Всего тестов: {total}")
        print(f"🌐 Адрес сервера: {self.base_url}")
        print("=" * 60)
        
        for i, test_case in enumerate(tests, 1):
            print(f"\n[{i}/{total}] Тест {test_case['id']}: {test_case['question'][:70]}...")
            result = await self.run_test(test_case)
            self.results.append(result)
            
            # Вывод результата
            status = "✅" if (result.passed_keywords or result.passed_confidence) else "❌"
            print(f"  {status} Уверенность: {result.confidence:.2f} | Время: {result.response_time:.2f}с")
            
            if result.actual_answer:
                # Очищаем ответ для отображения в консоли
                clean_answer = result.actual_answer.replace('\n', ' ').replace('\r', ' ')
                clean_answer = re.sub(r'\s+', ' ', clean_answer)
                preview = clean_answer[:200] + "..." if len(clean_answer) > 200 else clean_answer
                print(f"  📝 Ответ: {preview}")
            
            if result.error:
                print(f"  ❌ Ошибка: {result.error}")
            
            # Небольшая задержка между запросами
            await asyncio.sleep(0.5)
        
        self.print_summary()
        self.save_readable_results()
    
    def create_example_test_suite(self, filename: str):
        """Создает пример тестового набора"""
        example_suite = {
            "test_suite": {
                "name": "RAG System Basic Tests",
                "description": "Проверка основных возможностей RAG системы",
                "tests": [
                    {
                        "id": "TEST_001",
                        "question": "Что такое RAG?",
                        "expected_keywords": ["RAG", "генерация", "поиск"],
                        "min_confidence": 0.3,
                        "top_k": 3,
                        "expect_sources": True
                    },
                    {
                        "id": "TEST_002",
                        "question": "Как работает гибридный поиск?",
                        "expected_keywords": ["семантический", "лексический", "вектор"],
                        "min_confidence": 0.3,
                        "top_k": 3,
                        "expect_sources": True
                    },
                    {
                        "id": "TEST_003",
                        "question": "Для чего нужны эмбеддинги?",
                        "expected_keywords": ["вектор", "представление", "смысл"],
                        "min_confidence": 0.3,
                        "top_k": 3,
                        "expect_sources": True
                    },
                    {
                        "id": "TEST_999",
                        "question": "Вопрос которого нет в базе знаний",
                        "expected_keywords": ["найдена", "нет информации", "не знаю"],
                        "min_confidence": 0.0,
                        "top_k": 3,
                        "expect_sources": False
                    }
                ]
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(example_suite, f, ensure_ascii=False, indent=2)
    
    def print_summary(self):
        """Вывод сводки результатов"""
        total = len(self.results)
        if total == 0:
            print("\n❌ Нет результатов для отображения")
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
                print(f"  - {r.test_id}: {r.question[:50]}...")
        else:
            print(f"\n🎉 Все тесты пройдены!")
        
        # Сохраняем результаты в JSON
        self.save_results()
    
    def save_results(self, filename: str = "test_results.json"):
        """Сохранение результатов в JSON"""
        if not self.results:
            print("Нет результатов для сохранения")
            return
            
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
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
                    "sources_count": len(r.sources) if r.sources else 0,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены в {filename}")
    
    def save_readable_results(self, filename: str = "test_results.txt"):
        """Сохраняет результаты в читаемом формате"""
        if not self.results:
            print("Нет результатов для сохранения")
            return
            
        with open(filename, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write("=" * 80 + "\n")
            f.write(f"ОТЧЕТ О ТЕСТИРОВАНИИ RAG СИСТЕМЫ\n")
            f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Адрес сервера: {self.base_url}\n")
            f.write("=" * 80 + "\n\n")
            
            # Общая статистика
            total = len(self.results)
            passed_keywords = sum(1 for r in self.results if r.passed_keywords)
            passed_confidence = sum(1 for r in self.results if r.passed_confidence)
            avg_confidence = sum(r.confidence for r in self.results) / total
            avg_response_time = sum(r.response_time for r in self.results) / total
            
            f.write("📊 СТАТИСТИКА:\n")
            f.write(f"  Всего тестов: {total}\n")
            f.write(f"  Успешно по ключевым словам: {passed_keywords}/{total} ({passed_keywords/total*100:.1f}%)\n")
            f.write(f"  Успешно по уверенности: {passed_confidence}/{total} ({passed_confidence/total*100:.1f}%)\n")
            f.write(f"  Средняя уверенность: {avg_confidence:.2f}\n")
            f.write(f"  Среднее время ответа: {avg_response_time:.2f}с\n")
            f.write("\n" + "-" * 80 + "\n\n")
            
            # Детальные результаты
            f.write("📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n\n")
            
            for i, r in enumerate(self.results, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"ТЕСТ #{i} - {r.test_id}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Вопрос: {r.question}\n")
                f.write(f"Уверенность: {r.confidence:.2f}\n")
                f.write(f"Время ответа: {r.response_time:.2f}с\n")
                
                status_parts = []
                if r.passed_keywords:
                    status_parts.append("ключевые слова ✓")
                if r.passed_confidence:
                    status_parts.append("уверенность ✓")
                
                status_text = " | ".join(status_parts) if status_parts else "❌ ПРОВАЛЕН"
                f.write(f"Статус: {status_text}\n")
                
                # Очищаем ответ от \n для читаемости
                clean_answer = r.actual_answer.replace('\n', ' ').replace('\r', ' ')
                clean_answer = re.sub(r'\s+', ' ', clean_answer)
                
                f.write(f"\nОтвет:\n{clean_answer}\n")
                
                # Ключевые слова
                if r.expected_keywords:
                    f.write(f"\nОжидаемые ключевые слова: {', '.join(r.expected_keywords)}\n")
                    keyword_stats = self.check_all_keywords(r.actual_answer, r.expected_keywords)
                    if keyword_stats["found"]:
                        f.write(f"✅ Найдено: {', '.join(keyword_stats['found'])}\n")
                    if keyword_stats["missing"]:
                        f.write(f"❌ Не найдено: {', '.join(keyword_stats['missing'])}\n")
                
                # Источники
                if r.sources:
                    f.write(f"\n📚 Источники ({len(r.sources)}):\n")
                    for src in r.sources[:3]:  # Показываем первые 3
                        if isinstance(src, dict):
                            source_info = src.get("source", src.get("metadata", {}).get("source", "unknown"))
                            f.write(f"  - {source_info}\n")
                
                if r.error:
                    f.write(f"\n❌ Ошибка: {r.error}\n")
                
                f.write("\n")
        
        print(f"\n📄 Читаемый отчет сохранен в {filename}")

async def main():
    print("=" * 60)
    print("🧪 RAG System Tester")
    print("=" * 60)
    print("\nУбедитесь, что сервер запущен:")
    print("  docker-compose up -d  # если используете Docker")
    print("  или")
    print("  uvicorn src.main:app --reload  # если локально")
    print("\n" + "=" * 60 + "\n")
    
    async with RAGSystemTester() as tester:
        # Проверяем здоровье системы
        if not await tester.test_health():
            print("\n❌ Система не запущена!")
            print("\nЗапустите сервер в другом терминале:")
            print("  cd AMAZING-RAG")
            print("  conda activate amazing_rag")
            print("  uvicorn src.main:app --reload")
            print("\nИли через Docker:")
            print("  docker-compose up -d")
            return
        
        # Запускаем тесты
        await tester.run_suite("test_suite.json")

if __name__ == "__main__":
    asyncio.run(main())