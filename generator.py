from accelerate.utils.memory import clear_device_cache
from transformers import pipeline

clear_device_cache()


def generate_response(prompt, rag_response, model, tokenizer):

    messages = [
        {'role': 'system', 'content': 'Ты - интеллектуальный ассистент, который помогает с поступлением в '
                                      'Магнитогорский государственный технический университет имени Г. И. Носова. '
                                      '(МГТУ им. Г. И. Носова).\n'
                                      'Отвечай кратко, по существу и понятным языком. Не расписывай очевидные вещи. '
                                      'Избегай длинных рассуждений, говори только главное. '
                                      'Вопрос находится в тегах <question> </question>. '
                                      'Документы, основываясь на которых ты должен отвечать на вопрос, находятся '
                                      'в тегах <document> </document>.'
                                      'Ты разговариваешь на русском языке.'
                                      'Если в документах нет ответа, ты должен написать - "Извините, ответ не найден".'
                                      '**Никогда не выдумывай факты**. Если у тебя нет данных, скажи: *«Мне нужно уточнить этот вопрос. Обратитесь в приёмную комиссию»*.'},
        {'role': 'user', 'content': 'Какая стоимость обучения на прикладной информатике?'},
        {'role': 'assistant', 'content': 'На 2025 год стоимость обучения в МГТУ им. Г. И. Носова на '
                                         'направлении "09.03.03 Прикладная информатика" '
                                         '- 145.000 рублей за год обучения. '},
        {'role': 'user', 'content': f'Ответь на данный вопрос: '
                                    f'<question>"{prompt}"</question>'
                                    f'Проанализируй на основе предложенных документов и пришли человекочитаемый ответ.'
                                    f'Если в документе содержится ответ из базы FAQ, то ты не должен сильно его переделывать'
                                    f'<document>'
                                    f'{rag_response}'
                                    f'</document>'}
    ]

    #
    # inputs = tokenizer([text], return_tensors="pt").to(model.device)
    #
    # generated_ids = model.generate(
    #     **inputs,
    #     max_new_tokens=2048,
    #     pad_token_id=tokenizer.pad_token_id,
    #     early_stopping=True
    # )
    #
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    # ]
    #
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)[0]['generated_text']

    with open('ans.txt', 'w', encoding='utf-8') as file:
        file.write(str(output))
    print(output)
    return output

