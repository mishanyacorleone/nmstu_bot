import json
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# === Пути ===
INDEX_PATH = "data/all_faiss_index.idx"
META_PATH = "data/all_documents_meta.json"

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Новые тексты
source = 'пояснения'
text = 'СПО - среднее профессиональное образование. Образовательная программа, позволяющая выпускникам, получившим основное общее образование, получить профессию за короткий период (обычно 2-4 года). СПО - это обучение в колледже, например, МПК (Многопрофильный колледж при МГТУ им. Г.И. Носова).'
full_text = {"source": source, "text": text}

embedding = model.encode([full_text["text"]]).astype("float32")

# === Шаг 4: Добавляем в индекс и метаданные ===
index.add(embedding)
metadata.append(full_text)

# === Шаг 5: Сохраняем ===
faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ Добавлено в индекс и сохранено.")
