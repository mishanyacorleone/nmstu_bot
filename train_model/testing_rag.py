# from sentence_transformers import SentenceTransformer
# import numpy as np
# import onnxruntime as ort
#
#
# onnx_model_path = 'paraphrase-multilingual-MiniLM-L12-v2/onnx/model_O3.onnx'
#
# session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
#
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#
# sentences = ['Привет, как поступить в университет?', 'Какие есть правила приема?']
#
# inputs = model.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# input_ids = inputs['input_ids'].cpu().numpy().astype(np.int64)
# attention_mask = inputs['attention_mask'].cpu().numpy().astype(np.int64)
# token_type_ids = inputs["token_type_ids"].cpu().numpy().astype(np.int64)
#
# outputs = session.run(None, {
#     'input_ids': input_ids,
#     'attention_mask': attention_mask,
#     'token_type_ids': token_type_ids
# })
#
# embeddings = np.array(outputs[0])
#
# sentence_embeddings = np.mean(embeddings, axis=1)
#
# print(f'Финальный размер эмбеддингов: {sentence_embeddings.shape}')
# print(f'Пример усредненного эмбеддинга: {sentence_embeddings[0][:5]}')

'''
Вопрос для раздумья. Необходимо, чтобы модель анализировала документ только тогда, когда она уверена, что в данном документе
есть ответ на поставленный вопрос. Если она не уверена, то должна сказать, что не уверена или же попробовать сгенерировать ответ
самостоятельно. Например, если вопрос "На какое направление мне лучше всего поступить, если я сдавал [перечень предметов],
то она не должна искать это в документе, а попробовать сгенерировать это самостоятельно.
'''


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# 1. Загрузка модели
model = SentenceTransformer("../sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# 2. Чтение файла и разбиение на части
def load_text(file_path, chunk_size=500):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # chunks = text.split('\n')
    # chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    pattern = r"\d+\..*?(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    return matches


text_chunks = load_text("../data/pravila_priema_bsm_na_2025.txt")

# 3. Генерация эмбеддингов
embeddings = model.encode(text_chunks, convert_to_numpy=True)

# 4. Создание FAISS-индекса
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2-нормализованный поиск
index.add(embeddings)


# 5. Функция поиска
def search(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [text_chunks[i] for i in indices[0]]
    return results


# 6. Тестируем поиск
query = "Когда начинается прием документов?"
results = search(query)

print("🔍 Найденные фрагменты:")
for res in results:
    print(res, "\n---")

# from datasets import load_dataset, Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from transformers import BitsAndBytesConfig
# # from sklearn.model_selection import train_test_split
# from peft import get_peft_model, LoraConfig, TaskType, PeftModel
# from accelerate.utils.memory import clear_device_cache
# import torch
#
# clear_device_cache()
#
# # dataset = load_dataset('json', data_files='data.json')
#
# # print(dataset['train'][0])
#
# MODEL_NAME = r'RefalMachine_ruadapt_qwen2.5_3B_ext_u48_instruct'
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_compute_dtype=torch.float16,
# )
#
# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_NAME,
#     # use_fast=True,
#     local_files_only=True,
#     # legacy=False
# )
#
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=quantization_config,
#     local_files_only=True
# ).to(device)
#
# model = PeftModel.from_pretrained(model, 'results/checkpoint-14940')
#
# prompt = f'<system>Ты - интеллектуальный ассистент, который помогает с поступлением в университет.' \
#          f'Ответь на вопрос, используя информацию из документа. Если в документе нет ответа, напиши "Информации нет в документе"</system>' \
#          f'<rules>Правила:' \
#          f'1. Не придумывай информацию.' \
#          f'2. Не вводи в заблуждение.' \
#          f'3. Отвечай только на вопрос, который задал пользователь. ' \
#          f'4. Отвечай кратко. ' \
#          f'5. Отвечай своими словами, как ты бы отвечал школьнику. </rules>' \
#          f'' \
#          f'<question>Вопрос: {query}</question>' \
#          f'<document>Документ: {" ".join(results)}</document>' \
#          f'Ответ (без повторения вопроса):'
#
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
#
# output = model.generate(**inputs,
#                         max_new_tokens=512,
#                         num_beams=5,
#                         do_sample=True,
#                         top_p=0.85,
#                         temperature=0.9,
#                         early_stopping=True,
#                         repetition_penalty=2.0,
#                         # eos_token_id=tokenizer.eos_token_id
#                         )
# print(type(inputs))
# print(inputs)
#
# print(tokenizer.decode(output[:, inputs['input_ids'].shape[1]:][0], skip_special_tokens=True))