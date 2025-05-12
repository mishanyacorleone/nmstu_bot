import random
import torch.cuda
from fastapi import FastAPI, Request
import httpx
import uvicorn
from llama_cpp import Llama
from generator import generate_response
from rag import retrieve_documents
import os

app = FastAPI()

# Эти значения предоставлены Jivo
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.join(current_dir, 'RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct_gguf/Q4_K_M.gguf')
BOT_TOKEN = "Замените на свой токен Telegram Bot"
PROVIDER_ID = "Замените на свой Provider ID от Jivo"
JIVO_API_URL = f"https://bot.jivosite.com/webhooks/{PROVIDER_ID}/{BOT_TOKEN}"


@app.on_event("startup")
async def startup_event():
    global llm
    # Инициализируем модель только один раз при запуске сервера
    print(f"Загрузка модели из: {MODEL_NAME}")
    llm = Llama(
        model_path=MODEL_NAME,
        n_ctx=8192,          # Размер контекста
        n_gpu_layers=-1,     # Использовать все слои на GPU
        seed=42,
        verbose=False,
        n_threads=4
    )
    print("Модель успешно загружена!")


@app.post("/jivo-webhook/7605496089:AAE0-Xn_3jsw4yB0XZXzvPGGmyaf6x57Voc")
async def jivo_webhook(request: Request):
    data = await request.json()
    print("📥 Получено из Jivo:", data)

    event_type = data.get("event")

    # Обрабатываем только сообщения от клиента
    if event_type != "CLIENT_MESSAGE":
        return {"status": "ignored", "reason": f"Событие {event_type} не требует обработки"}

    # Извлечение текста сообщения и ID чата
    user_message = data.get("message", {}).get("text", "")
    chat_id = data.get("chat_id")
    client_id = data.get('client_id')

    if user_message and chat_id:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Ответ от модели
        retrieved_docs = retrieve_documents(user_message) # Поиск документов

        rag_response = ''
        for elem in retrieved_docs:
            print(f"С вероятностью: {elem['score']}\n"
                  f"Ответ содержится в следующем чанке (corpus_id={elem['corpus_id']}):\n"
                  f"{elem['candidate']}\n\n")

            rag_response += f"{elem['candidate']}\n\n"
        try:
            response = generate_response(user_message, rag_response, llm)
        except Exception as ex:
            print(f"Ошибка при генерации ответа: {ex}")
            response = "Извините, произошла ошибка при обработке вашего запроса. Попробуйте переформулировать вопрос или обратитесь позже."

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Формируем ответ Jivo
        payload = {
            'id': str(random.randint(1, 10000000)),
            "client_id": client_id,  # Client ID из события CLIENT_MESSAGE
            'chat_id': chat_id, # Chat ID из события CLIENT_MESSAGE
            "message": {
                "type": "TEXT",
                "text": response,
            },
            'event': 'BOT_MESSAGE'
        }

        # Отправка ответа в Jivo
        async with httpx.AsyncClient() as client:
            jivo_response = await client.post(JIVO_API_URL,
                                              json=payload,
                                              headers={"Content-Type": "application/json"})
            print("📤 Ответ отправлен в Jivo:", jivo_response.text)

    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)