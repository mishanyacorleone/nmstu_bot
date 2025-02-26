from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig, TaskType
import torch

# 📌 Загружаем датасет
dataset = load_dataset('json', data_files='answer-response.json')

MODEL_NAME = r'RefalMachine_ruadapt_qwen2.5_3B_ext_u48_instruct'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 📌 Настройка 4-битной квантованной модели
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)

# 📌 Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

# 📌 Загружаем модель
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    local_files_only=True
).to(device)

# # 🔥 Замораживаем ВСЕ параметры модели
# for param in model.parameters():
#     param.requires_grad = False
#
# # 🔥 Размораживаем только последние 4 слоя
# for layer in model.model.layers[-4:]:
#     for param in layer.parameters():
#         param.data = param.data.to(torch.float16)  # Приводим к float16
#         param.requires_grad = True

# 📌 LoRA-конфигурация
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj']
)

# 📌 Подключаем LoRA к модели
model = get_peft_model(model, peft_config)


# 📌 Функция токенизации
def tokenize_function(batch):
    texts = [p + " " + r for p, r in zip(batch["prompt"], batch["response"])]
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    encodings['labels'] = encodings['input_ids']
    return encodings


# 📌 Токенизируем датасет
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 📌 Разделяем на train/test
dataset_split = tokenized_datasets['train'].train_test_split(test_size=0.1)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

# 📌 Настройки обучения
training_args = TrainingArguments(
    fp16=True,
    output_dir="./results",
    gradient_checkpointing=False,
    evaluation_strategy="steps",
    eval_steps=1000,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-5,
    num_train_epochs=3,
)

# 📌 Trainer для обучения
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 🚀 Запускаем обучение
trainer.train()

# 📌 Сохраняем модель и токенизатор
model.save_pretrained('nmstuModal-based-qwen2.5-3B-v0.2')
tokenizer.save_pretrained('nmstuModal-based-qwen2.5-3B-v0.2')

# 📌 Тестируем генерацию
prompt = '/start'
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output = model.generate(**inputs,
                        max_length=128,
                        do_sample=True,
                        early_stopping=True
                        )

print(tokenizer.decode(output[0], skip_special_tokens=True))

