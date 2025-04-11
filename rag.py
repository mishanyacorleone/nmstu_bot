import json
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Загружаем модель sentence-transformers
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Функция загрузки FAISS-индекса и соответствующих текстов
def load_faiss_index(index_path, texts_path):
    index = faiss.read_index(index_path)
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    return index, texts

# Загружаем индекс и тексты
pravila_index, pravila_texts = load_faiss_index("data/faiss_index_pravila.idx", "data/pravila_chunks.json")
specs_index, specs_texts = load_faiss_index("data/faiss_index_specs.idx", "data/spec_chunks.json")

# Поиск в выбранном индексе
def retrieve_from_index(query, index, texts, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(score),
            "corpus_id": idx,
            "candidate": texts[idx]
        })
    return results

# Главная функция — запуск поиска
def retrieve_documents(query, source="specs", top_k=5):
    if source == "pravila":
        return retrieve_from_index(query, pravila_index, pravila_texts, top_k)
    elif source == "specs":
        return retrieve_from_index(query, specs_index, specs_texts, top_k)
    else:
        raise ValueError("Неверный источник. Используй 'pravila' или 'specs'.")

# # Пример запуска
# if __name__ == "__main__":
#     query = "В какие сроки осуществляется прием заявлений?"
#     results = retrieve_documents(query, source="pravila", top_k=5)
#
#     for res in results:
#         print(f"Score: {res['score']:.4f}")
#         print(res['candidate'])
#         print("-" * 80)
