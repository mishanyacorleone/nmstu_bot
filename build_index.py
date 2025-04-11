# build_index.py
import json
import re
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Инициализируем модель
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

### 1. Загружаем правила приема (TXT) ###
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = r"\d+\..*?(?=\n\d+\.|\Z)"  # Разбиение текста на блоки по номерам (1. ..., 2. ...)
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

pravila_texts = load_text("data/pravila_priema_bsm_na_2025.txt")

### 2. Загружаем специальности (JSON) ###
with open("data/specs_magtu.json", "r", encoding="utf-8") as f:
    specs_data = json.load(f)

specs_texts = []
spec_index = {}

for idx, spec in enumerate(specs_data):
    spec_text = f"{spec['spec_name']} | {spec['about']} | {spec['profile_name']} | " \
                f"{json.dumps(spec['education_data'], ensure_ascii=False)} | " \
                f"{json.dumps(spec['exams'], ensure_ascii=False)} | " \
                f"{spec['profile_skills']} | {spec['profile_faculty']} | {spec['professions']}"
    specs_texts.append(spec_text)
    spec_index[idx] = spec

# Получаем эмбеддинги
pravila_embeddings = model.encode(pravila_texts, convert_to_numpy=True, show_progress_bar=True)
specs_embeddings = model.encode(specs_texts, convert_to_numpy=True, show_progress_bar=True)

# Создаём FAISS индексы
index_pravila = faiss.IndexFlatL2(pravila_embeddings.shape[1])
index_specs = faiss.IndexFlatL2(specs_embeddings.shape[1])

index_pravila.add(pravila_embeddings)
index_specs.add(specs_embeddings)

# Сохраняем индексы и документы
faiss.write_index(index_pravila, "faiss_index_pravila.idx")
faiss.write_index(index_specs, "faiss_index_specs.idx")

with open("data/pravila_chunks.json", "w", encoding="utf-8") as f:
    json.dump(pravila_texts, f, ensure_ascii=False, indent=2)

with open("data/spec_chunks.json", "w", encoding="utf-8") as f:
    json.dump(specs_texts, f, ensure_ascii=False, indent=2)
