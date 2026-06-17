import asyncio
import aiohttp
import json

async def test():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://127.0.0.1:8000/query",
            json={"question": "Где можно скачать лаунчер для игры?"}
        ) as resp:
            result = await resp.json()
            
            # Выводим "сырой" JSON
            print("RAW JSON:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            print("\n" + "="*50)
            print("ОТФОРМАТИРОВАННЫЙ ВЫВОД:")
            print("="*50)
            # Здесь \n будут отображаться как реальные переносы
            print(result['answer'])

asyncio.run(test())