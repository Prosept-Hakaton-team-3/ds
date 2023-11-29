## Описание проекта

Заказчик производит несколько сотен различных товаров бытовой и промышленной химии, а затем продаёт эти товары через дилеров. Дилеры, в свою очередь, занимаются розничной продажей товаров в крупных сетях магазинов и на онлайн площадках.
Для оценки ситуации,  управления ценами и  бизнесом в целом, заказчик периодически собирает информацию о том, как дилеры продают их товар. Для этого они парсят сайты дилеров, а затем сопоставляют товары и цены.
Зачастую описание товаров на сайтах дилеров отличаются от того описания, что даёт заказчик. 

**Цель проекта** - разработка решения, которое отчасти автоматизирует процесс сопоставления товаров. Основная идея - предлагать несколько товаров заказчика, которые с наибольшей вероятностью соответствуют размечаемому товару дилера. 
Предлагается реализовать это решение, как онлайн сервис, открываемый в веб-браузере. Выбор наиболее вероятных подсказок делается методами машинного обучения.


## Распределение задачей в команде

**Анатолий Остапенко** - разработка baseline и функции унификации наименований, коммуникация с наставниками и общая координация

**Владимир Зотов** - подбор оптимальной модели для создания эмбеддингов, оптимизация функции унификации наименований, проведение экспериментов с иными параметрами

**Александр Диков** - подбор оптимальной модели для создания эмбеддингов, упаковка модели для backend-разработчиков, проведение экспериментов с иными параметрами

## Описание проделанной работы

1. Проведено исследование таблиц и их объединение по ключам.
2. Разработана функция для унификации наименований товаров на сайте Заказчика и на сайтах дилеров с использованием регулярных выражений.
3. Разработан baseline модели, который предусматривал:
   - создание эмбеддингов при помощи модели bert-base-multilingual-cased;
   - расчет косинусного расстояния между эмбеддингом наименования каждого продукта с сайта дилера и всеми уникальными наименованиями продуктов с сайта заказчика при помощи метода pdist() библиотеки scipy и ранжирование эмбеддингов в зависимости от рассчитанного расстояния;
   - предложение 5 вариантов продуктов с сайта заказчика с наименьшим косинусным расстоянием до целевого продукта с сайта дилера.
4. Оптимизация baseline путем подбора оптимальной модели для создания эмбеддингов (в итоге остановились на Sentence Transformer LaBSE (Language-agnostic BERT Sentence Embedding)) и доработки функции унификации наименований.
5. Упаковка решения в скрипт для backend-разработчиков.

## Используемая метрика качества

В качестве метрики использовали Recall at K, поскольку целью проекта является разработка ML-алгоритма, среди предлагаемых вариантов сопоставления которого всегда должен содержаться правильный вариант для разметки (независимо от количества предлагаемых вариантов для разметки).

## Инструкция для использования решения

#### Описание класса ProseptDescriptionSearcher

Класс `ProseptDescriptionSearcher` предназначен для поиска продукта в базе данных по описанию. Он использует `SentenceTransformer` для генерации векторных представлений описаний продуктов и выполняет поиск на основе косинусного расстояния.

#### Инициализация

Параметры:
1. `data_connect_or_path`: обязательный параметр. Отвечает за получения данных. Подключение к базе данных или путь для загрузки данных из CSV-файла;
2. `mode`: Чтение из базы данных `SQL` или `CSV`-файла. Варианты: `'csv'` или `None`. `None` - дефолтное значение, подключение к БД;
3. `model_name_or_path`: Путь к сохраненной модели `SentenceTransformer`;
4. `number_of_matching`: Количество вариантов сопоставления;
5. `cache_embeddings_update`: Флаг для сохранения векторных представлений продуктов. `False` - дефолтное значение. Стоит активировать, если данные изменились.

Инициализация:

1. Загружается модель;
2. Определяются основные параметры;
3. Загружаются данные в формате `БД` или `csv`;
4. Выполняются функции отчистки, соединение всей информации в один df;
5. Генерация embeddings;

Использование модели:

* Получение вариантов сопоставления обеспечивает функция `make_predict`:

    Параметры:
    `descriptions_to_match`: Список описаний для поиска. 
    `number_of_matching`: Количество вариантов сопоставления.

    Return:
    `dict`: Формат {'Запрос' : {id_1 : наименование_товара, id_2 : наименование_товара}}
    Ранжирует продукты на основе схожести и возвращает словарь с соответствующими идентификаторами и именами продуктов.

* Получение качества модели путём расчёта метрики обеспечивает `calculate_recall_at_n`:

    Создает аттрибут класса `average_recall`, который представляет собой числовую оценку модели.

Пример:

```python
from prosept_product import ProseptDescriptionSearcher

# Инициализация объекта ProseptDescriptionSearcher
prosept = ProseptDescriptionSearcher(data_connect_or_path=data_path, mode='csv')

# Расчет recall для топ-N рекомендаций (по умолчанию N=5)
prosept.calculate_recall_at_n(number_of_matching=5)

# Make predictions based on descriptions
descriptions_to_match = ["Description 1", "Description 2", ...]

# Получение результатов предсказания в виде словаря
predictions = description_searcher.make_predict(descriptions_to_match)
```

