import asyncio
import json
import re
import random
import os
import time
import numpy as np
from collections import deque
import hashlib
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import torch.cuda
import uvicorn

# ML модели
import faiss
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder

# Мои зависимости
from generator import generate_response
from rag import retrieve_documents

# Настройки
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.join(current_dir, 'RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct_gguf/Q4_K_M.gguf')
BOT_TOKEN = "7845430145:AAEgWOV4ru46OtYZOo4b119jRFIgmWz4GaI"
PROVIDER_ID = "MPGhfH1e1VjxxmK"
JIVO_API_URL = f"https://bot.jivosite.com/webhooks/{PROVIDER_ID}/{BOT_TOKEN}"

# FAISS настройки
INDEX_PATH = "data/all_faiss_index.idx"
META_PATH = "data/all_documents_meta.json"

# Дедупликация сообщений
processed_messages = {}  # message_id -> timestamp
MESSAGE_EXPIRY_MINUTES = 30  # Время жизни записи о сообщении


# Модели для API запросов
class DocumentData(BaseModel):
    source: str
    text: str
    summarize: str


class UpdateResponse(BaseModel):
    success: bool
    message: str
    total_documents: int


app = FastAPI()

llm, retriever_model, reranker_model = None, None, None
faiss_index = None
metadata = None

# Очередь для последовательной обработки сообщений
message_queue = deque()
processing_lock = asyncio.Lock()
is_processing = False


def generate_message_hash(client_id: str, chat_id: str, message_text: str, timestamp: int) -> str:
    """
    Генерирует уникальный хеш для сообщения на основе его содержимого
    Используется для дедупликации повторных запросов от Jivo
    """
    # Округляем timestamp до 10 секунд для учета небольших расхождений времени
    rounded_timestamp = (timestamp // 10) * 10
    message_data = f"{client_id}:{chat_id}:{message_text}:{rounded_timestamp}"
    return hashlib.md5(message_data.encode()).hexdigest()


def is_message_processed(message_hash: str) -> bool:
    """
    Проверяет, было ли сообщение уже обработано
    """
    global processed_messages

    # Очищаем устаревшие записи
    current_time = datetime.now()
    expired_hashes = []

    for msg_hash, timestamp in processed_messages.items():
        if current_time - timestamp > timedelta(minutes=MESSAGE_EXPIRY_MINUTES):
            expired_hashes.append(msg_hash)

    for expired_hash in expired_hashes:
        del processed_messages[expired_hash]

    return message_hash in processed_messages


def mark_message_as_processed(message_hash: str):
    """
    Отмечает сообщение как обработанное
    """
    global processed_messages
    processed_messages[message_hash] = datetime.now()


@app.on_event("startup")
async def startup_event():
    global llm, retriever_model, reranker_model, faiss_index, metadata
    # Инициализируем модель только один раз при запуске сервера
    print(f"Загрузка модели из: {MODEL_NAME}")
    llm = Llama(
        model_path=MODEL_NAME,
        n_ctx=6144,  # Размер контекста
        n_gpu_layers=20,  # Использовать все слои на GPU
        seed=42,
        verbose=True,
        repet_penalty=1.2,
        n_threads=4,
        offload_kqv=True,
    )

    retriever_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    reranker_model = CrossEncoder("DiTy/cross-encoder-russian-msmarco", max_length=512, device='cuda')

    print("Модель успешно загружена!")

    try:
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
        print(f"✅ FAISS индекс загружен. Документов: {len(metadata)}")
    except Exception as ex:
        print(f"❌ Ошибка загрузки FAISS индекса: {ex}")
        faiss_index = None
        metadata = []

    # Запускаем обработчик очереди
    asyncio.create_task(message_processor())


async def message_processor():
    """Обработчик очереди сообщений - работает последовательно"""
    global message_queue, is_processing

    while True:
        async with processing_lock:
            if message_queue and not is_processing:
                is_processing = True
                data = message_queue.popleft()
                print(f"🔄 Обрабатываю сообщение из очереди. Осталось в очереди: {len(message_queue)}")

                try:
                    await process_message_sync(data)
                except Exception as e:
                    print(f"❌ Ошибка при обработке сообщения: {e}")
                finally:
                    is_processing = False

        # Небольшая пауза перед следующей проверкой очереди
        await asyncio.sleep(0.1)


async def save_faiss_data():
    """Сохранение FAISS индекса и метаданных на диск"""
    global faiss_index, metadata

    try:
        # Сохраняем индекс
        faiss.write_index(faiss_index, INDEX_PATH)

        # Сохраняем метаданные
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения FAISS данных: {e}")
        return False


@app.post("/update-index", response_model=UpdateResponse)
async def update_faiss_index(document: DocumentData):
    """
    Обновление FAISS индекса новым документом
    """
    global faiss_index, metadata, retriever_model

    if faiss_index is None or metadata is None:
        raise HTTPException(status_code=500, detail="FAISS индекс не инициализирован")

    if retriever_model is None:
        raise HTTPException(status_code=500, detail="Модель эмбеддингов не загружена")

    try:
        # Подготавливаем данные документа
        full_text = {
            "source": document.source,
            "text": document.text,
            "summarize": document.summarize
        }

        # Генерируем эмбеддинг
        embedding = retriever_model.encode([full_text["summarize"]]).astype("float32")

        # Добавляем в индекс
        faiss_index.add(embedding)

        # Добавляем в метаданные
        metadata.append(full_text)

        # Сохраняем на диск
        save_success = await save_faiss_data()

        if save_success:
            print(f"✅ Добавлен новый документ. Всего документов: {len(metadata)}")
            return UpdateResponse(
                success=True,
                message="Документ успешно добавлен в индекс",
                total_documents=len(metadata)
            )
        else:
            # Откатываем изменения в памяти, если не удалось сохранить
            faiss_index.remove_ids(np.array([len(metadata) - 1]))
            metadata.pop()
            raise HTTPException(status_code=500, detail="Не удалось сохранить данные на диск")

    except Exception as e:
        print(f"❌ Ошибка обновления индекса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обновления индекса: {str(e)}")


@app.get("/index-stats")
async def get_index_stats():
    """Получение статистики по индексу"""
    global faiss_index, metadata

    if faiss_index is None or metadata is None:
        return {"error": "FAISS индекс не инициализирован"}

    return {
        "total_documents": len(metadata),
        "index_size": faiss_index.ntotal,
        "index_dimension": faiss_index.d
    }


@app.get("/queue-stats")
async def get_queue_stats():
    """Получение статистики очереди сообщений"""
    return {
        "queue_length": len(message_queue),
        "is_processing": is_processing,
        "processed_messages_count": len(processed_messages)
    }


async def send_message_to_jivo(message_id, client_id, chat_id, text):
    payload = {
        "id": message_id,
        "client_id": client_id,  # Client ID из события CLIENT_MESSAGE
        "chat_id": chat_id,  # Chat ID из события CLIENT_MESSAGE
        "message": {
            "type": "MARKDOWN",
            "content": text,
            "text": text
        },
        "event": 'BOT_MESSAGE',
        "timestamp": int(time.time())
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(JIVO_API_URL,
                                         json=payload,
                                         headers={"Content-Type": "application/json"}
                                         )
            print(f'✅ Отправлено в Jivo: {response.status_code}')
    except Exception as ex:
        print(f'❌ Ошибка отправки в Jivo: {ex}')


async def generate_full_answer(user_message):
    """Генерация полного ответа с использованием RAG"""
    retrieved_docs = await asyncio.to_thread(retrieve_documents, user_message, retriever_model,
                                             reranker_model, faiss_index, metadata)

    rag_response = ''
    for elem in retrieved_docs:
        print(f"С вероятностью: {elem['score']}\n"
              f"Ответ содержится в следующем чанке (corpus_id={elem['corpus_id']}):\n"
              f"{elem['candidate']}\n\n")

        rag_response += f"{elem['candidate']}\n\n"

    try:
        response = await asyncio.to_thread(generate_response, user_message, rag_response, llm)
    except Exception as ex:
        print(f"❌ Ошибка при генерации ответа: {ex}")
        response = "Извините, произошла ошибка при обработке вашего запроса. Попробуйте переформулировать вопрос или обратитесь позже."

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response


async def transfer_to_operator(message_id, client_id, chat_id):
    payload = {
        'id': message_id,
        "client_id": client_id,  # Client ID из события CLIENT_MESSAGE
        'chat_id': chat_id,  # Chat ID из события CLIENT_MESSAGE
        'event': 'INVITE_AGENT'
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(JIVO_API_URL,
                                         json=payload,
                                         headers={"Content-Type": "application/json"}
                                         )
            print(f'✅ Оператор приглашен: {response.status_code}')
    except Exception as ex:
        print(f'❌ Ошибка приглашения оператора: {ex}')


async def process_message_sync(data: dict):
    """Синхронная обработка сообщения (выполняется последовательно)"""
    global llm, retriever_model, reranker_model

    # Извлечение текста сообщения и ID чата
    user_message = data.get("message", {}).get("text", "")
    chat_id = data.get("chat_id")
    client_id = data.get('client_id')

    if not user_message or not chat_id:
        print("❌ Пустое сообщение или отсутствует chat_id")
        return

    print(f"🤖 Обрабатываю сообщение: {user_message[:50]}...")

    if user_message == '/start':
        response = ("Привет! Я интеллектуальный помощник по поступлению в МГТУ им. Г. И. Носова. "
                    "Ты можешь задать мне любой вопрос: какие экзамены нужны, какие проходные баллы и многое другое. "
                    "Если я не смогу помочь — просто напиши 'оператор', и я передам диалог человеку 😊")

    elif user_message.lower() in ['/operator', 'оператор']:
        is_agents_online = data.get("agents_online")
        if is_agents_online:
            response = "Очень жаль, что я не смог вам помочь. Перевожу разговор на оператора!"
            await transfer_to_operator(str(random.randint(1, 1000000)), client_id, chat_id)
        else:
            response = 'Извините, к сожалению, сейчас нет доступных операторов. Может быть я могу вам чем-нибудь помочь?\n' \
                       'Просто напишите свой вопрос, а я постараюсь на него ответить.'

    else:
        response = await generate_full_answer(user_message=user_message)

    await send_message_to_jivo(str(random.randint(1, 1000000)), client_id, chat_id, response)
    print(f"✅ Сообщение обработано и отправлено")


@app.post(f"/jivo-webhook/{BOT_TOKEN}")
async def jivo_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Webhook для получения сообщений от Jivo
    Сначала возвращает 200, потом обрабатывает сообщение
    """
    data = await request.json()
    print("📥 Получено из Jivo:", data)

    if data.get("event") == "CLIENT_MESSAGE":
        # Добавляем обработку в фоновые задачи
        background_tasks.add_task(add_message_to_queue, data)
        print("📝 Сообщение добавлено в фоновую обработку")

    # Немедленно возвращаем успешный ответ
    return JSONResponse(status_code=200, content={"ok": True})


async def add_message_to_queue(data: dict):
    """
    Добавляет сообщение в очередь обработки
    Выполняется в фоновой задаче
    """
    async with processing_lock:
        message_queue.append(data)
        print(f"📝 Сообщение добавлено в очередь. Всего в очереди: {len(message_queue)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)