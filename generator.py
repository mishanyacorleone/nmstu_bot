from accelerate.utils.memory import clear_device_cache

clear_device_cache()


def generate_response(prompt, rag_response, model, tokenizer):

    messages = [
        {'role': 'system', 'content': 'Ты - интеллектуальный ассистент, который помогает с поступлением в'
                                      'Магнитогорский государственный технический университет имени Г. И. Носова. '
                                      '(МГТУ им. Г. И. Носова).'
                                      # 'Вопрос находится в тегах <question> </question>.'
                                      # 'Документы, основываясь на которых ты должен отвечать на вопрос, находятся'
                                      # 'в тегах <document> </document>'
                                      'Ты разговариваешь на русском языке.'},
                                      # 'Если ответа на вопрос нет в документе, ты должен написать: '
                                      # '"Извините, я не знаю, но я могу позвать оператора."'},
        {'role': 'user', 'content': f'Ответь на данный вопрос: '
                                    f'<question>"{prompt}"</question>'
                                    f'Проанализируй на основе предложенных документов и пришли человекочитаемый ответ.'
                                    f'Если в документе содержится ответ из базы FAQ, то ты не должен сильно его переделывать'
                                    f'<document>'
                                    f'{rag_response}'
                                    f'</document>'}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,

    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response