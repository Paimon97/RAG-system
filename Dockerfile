# Dockerfile.conda
FROM continuumio/miniconda3:latest

WORKDIR /app

# Копируем environment.yml
COPY environment.yml .

# Создаем окружение из environment.yml
RUN conda env create -f environment.yml

# Активируем окружение
SHELL ["conda", "run", "-n", "amazing_rag", "/bin/bash", "-c"]

# Копируем код
COPY . .

# Устанавливаем спаси модель
RUN conda run -n amazing_rag python -m spacy download ru_core_news_sm

# Создаем директории для данных
# RUN mkdir -p /app/data/documents

# Открываем порт
EXPOSE 8000

# Команда для запуска
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]