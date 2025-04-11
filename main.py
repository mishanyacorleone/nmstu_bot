import asyncio
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from generator import generate_response
from rag import retrieve_documents
# from rag import get_response
import json


MODEL_NAME = r'../RefalMachine_ruadapt_qwen2.5_3B_ext_u48_instruct'
TOKEN = ""
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode='HTML'))
dp = Dispatcher()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    local_files_only=True
).to(device)


@dp.message(Command('start'))
async def start_message(message: Message):
    await message.answer('Привет! Я интеллектуальный помощник для помощи с поступлением в МГТУ! Ты можешь задать '
                         'мне любой вопрос по поступлению и я на него отвечу. Если я не знаю ответа, '
                         'ты можешь позвать оператора')


@dp.message(Command('operator'))
async def start_message(message: Message):
    await message.answer('Зову оператора!')


@dp.message()
async def handle_message(message: Message, model=model, tokenizer=tokenizer):
    user_query = message.text

    await bot.send_chat_action(message.chat.id, 'typing')

    retrieved_docs = await asyncio.to_thread(retrieve_documents, user_query, 3)
    # print(type(retrieved_docs))
    # print(retrieved_docs)
    # rag_response = '\n'.join([
    #     json.dumps(doc, ensure_ascii=False, indent=2) if isinstance(doc, dict) else str(doc)
    #     for doc in retrieved_docs
    # ])
    rag_response = str(retrieved_docs)
    # print(rag_response)

    response = await asyncio.to_thread(generate_response, user_query, rag_response, model, tokenizer)

    await message.answer(response)


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
