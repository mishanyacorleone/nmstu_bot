import json
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Загружаем модели
retriever_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
reranker_model = CrossEncoder("DiTy/cross-encoder-russian-msmarco", max_length=512)

# Загрузка FAISS-индекса и текстов

def load_faiss_index(index_path, texts_path):
    index = faiss.read_index(index_path)
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    return index, texts

# Загружаем объединенный индекс и тексты
index, all_documents = load_faiss_index("data/all_faiss_index.idx", "data/all_documents_meta.json")

# Функция первичного поиска
def retrieve_from_index(query, index, texts, top_k=10):
    query_embedding = retriever_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(score),
            "corpus_id": idx,
            "candidate": texts[idx]
        })
    return results

# Функция реранка
def rerank_documents(query, candidates, top_k=3):
    # Убедимся, что candidate — это строка
    pairs = []
    for doc in candidates:
        candidate_text = doc["candidate"]
        if isinstance(candidate_text, dict):
            candidate_text = json.dumps(candidate_text, ensure_ascii=False)
        elif not isinstance(candidate_text, str):
            candidate_text = str(candidate_text)

        pairs.append([query, candidate_text])

    # Получаем оценки от CrossEncoder
    scores = reranker_model.predict(pairs)

    # Объединяем оценки с кандидатами
    reranked = []
    for score, doc in zip(scores, candidates):
        reranked.append({
            "score": float(score),
            "corpus_id": doc["corpus_id"],
            "candidate": doc["candidate"]
        })

    # Сортировка по убыванию
    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

    return reranked[:top_k]

# Главная функция

def retrieve_documents(query, top_k=3):
    initial_candidates = retrieve_from_index(query, index, all_documents, top_k=10)
    return rerank_documents(query, initial_candidates, top_k=top_k)

# # Пример использования
# if __name__ == "__main__":
#     query = "Какие документы нужны для поступления?"
#     results = retrieve_documents(query, top_k=3)
#
#     for res in results:
#         print(f"Score: {res['score']:.4f}")
#         print(res['candidate'])
#         print("-" * 80)
