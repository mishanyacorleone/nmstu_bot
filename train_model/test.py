# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model_name = "../deepseek-ai DeepSeek-R1-Distill-Qwen-1.5B"  # Замени на свою модель
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
#
# messages = [
#         {'role': 'system', 'content': 'Ты - интеллектуальный помощник, разговаривающий на русском языке'},
#         {'role': 'user', 'content': 'Привет, как дела?'}
#     ]
#
# text = tokenizer.apply_chat_template(messages)
# print(text)
# inputs = tokenizer(str(text), return_tensors="pt").to('cuda')
#
# output = model.generate(**inputs, max_length=50)
# print(tokenizer.decode(output[0], skip_special_tokens=True))

# Load model directly
from transformers import pipeline

messages = [
    {'role': 'user', 'content': 'Привет, с чем ты мне можешь помочь? Отвечай только на русском языке'}
]

pipe = pipeline('text-generation', model='../deepseek-ai DeepSeek-R1-Distill-Qwen-1.5B', max_new_tokens=1024)

response = pipe(messages)
print(response)