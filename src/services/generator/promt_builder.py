
class PromptBuilder:

    @staticmethod
    def build_qa_prompt(
        query: str,
        contexts: list[str]
    ) -> str:

        joined_context = "\n\n".join(contexts)

        return f"""
Ты AI assistant для документации.

Правила:
1. Используй только контекст.
2. Не придумывай информацию.
3. Если ответа нет — ответь:
   "Информация не найдена".
4. Будь кратким.

Контекст:
{joined_context}

Вопрос:
{query}

Ответ:
"""