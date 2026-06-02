1. Создать conda-окружение с Python 3.12
conda create -n amazing_rag python=3.12

2. Активировать окружение
conda activate amazing_rag

3. Установить spaCy через conda
conda install -c conda-forge spacy=3.8.2
python -m spacy download ru_core_web_sm
conda install scikit-learn

4. Установить все остальные зависимости через pip
pip install -r requirements.txt

5.  Запустите контейнеры

Запуск Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

Запуск Elasticsearch
docker run -d -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:9.4.0

Запуск Redis
docker run -d -p 6379:6379 redis:7-alpine

6. Запуск
python -m src.main

7. Загрузка 
загрузка через api /documents/upload
файл с расширением .txt
