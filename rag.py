import json
import faiss
import re
from sentence_transformers import SentenceTransformer

# Инициализируем модель
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


### 1. Загружаем правила приема (TXT) ###
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = r"\d+\..*?(?=\n\d+\.|\Z)"  # Разбиение текста на блоки по номерам (1. ..., 2. ...)
    matches = re.findall(pattern, text, re.DOTALL)
    return matches  # Список блоков текста


pravila_texts = load_text("data/pravila_priema_bsm_na_2025.txt")

### 2. Загружаем специальности (JSON) ###
with open("data/specs_magtu.json", "r", encoding="utf-8") as f:
    specs_data = json.load(f)

# Объединяем данные о специальностях в строки
specs_texts = []
spec_index = {}

for idx, spec in enumerate(specs_data):
    spec_text = f"{spec['spec_name']} | {spec['about']} | {spec['profile_name']} | " \
                f"{json.dumps(spec['education_data'], ensure_ascii=False)} | " \
                f"{json.dumps(spec['exams'], ensure_ascii=False)} | " \
                f"{spec['profile_skills']} | {spec['profile_faculty']} | {spec['professions']}"
    specs_texts.append(spec_text)
    spec_index[idx] = spec  # Запоминаем соответствие индекса и специальности

### 3. Векторизуем тексты ###
pravila_embeddings = model.encode(pravila_texts, convert_to_numpy=True)
specs_embeddings = model.encode(specs_texts, convert_to_numpy=True)

# Создаем FAISS-индексы
dimension = pravila_embeddings.shape[1]
pravila_index = faiss.IndexFlatL2(dimension)
pravila_index.add(pravila_embeddings)

specs_index = faiss.IndexFlatL2(dimension)
specs_index.add(specs_embeddings)


### 4. Функция выбора источника данных ###
def classify_query(query):
    # Простейшая логика (можно заменить на классификатор)
    keywords_pravila = ["поступление", "документы", "баллы", "конкурс", "прием", "льготы"]
    keywords_specs = ["специальность", "обучение", "экзамены", "стоимость", "профиль", "кем стать"]

    for kw in keywords_pravila:
        if kw in query.lower():
            return "pravila"

    for kw in keywords_specs:
        if kw in query.lower():
            return "specs"

    return "specs"  # По умолчанию ищем в специальностях


### 5. Функция поиска ###
def retrieve_documents(query, top_k=3):
    source = classify_query(query)  # Определяем источник данных
    query_embedding = model.encode([query], convert_to_numpy=True)

    if source == "pravila":
        distances, indices = pravila_index.search(query_embedding, top_k)
        results = [pravila_texts[i] for i in indices[0]]
    else:
        distances, indices = specs_index.search(query_embedding, top_k)
        results = [spec_index[i] for i in indices[0]]

    return results