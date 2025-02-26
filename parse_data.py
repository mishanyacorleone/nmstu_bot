import json
# import csv
#
# path = str(input('Путь к данным: '))
#
# with open(path, "r", encoding="utf-8") as file:
#     data = json.load(file)
#
# clients = data['clients']
#
# formatted_data = []
#
# for client in clients:
#     client_id = client['client_id']
#     name = client['name']
#     chats = client['chats']
#     formatted_chats = []
#     if type(chats) == list:
#         for chat in chats:
#             chat_id = chat['chat_id']
#             events = chat['events']
#             messages = []
#
#             for event in events:
#                 if event['event_type'] == 'message':
#                     if event['agent_id'] is None:
#                         user_message = event['text']
#                         timestamp = event['created_ts']
#                         messages.append({'role': 'user', 'timestamp': timestamp, 'prompt': user_message})
#                     elif event['agent_id'] is not None:
#                         operator_response = event['text']
#                         timestamp = event['created_ts']
#                         messages.append({'role': 'operator', 'timestamp': timestamp, 'prompt': operator_response})
#
#             formatted_chats.append({
#                 "chat_id": chat_id,
#                 "messages": sorted(messages, key=lambda x: x["timestamp"])  # Сортируем по времени
#             })
#
#         formatted_data.append({
#             "client_id": client_id,
#             "chats": formatted_chats
#         })
#     else:
#         for key, chat in chats.items():
#             chat_id = chat['chat_id']
#             events = chat['events']
#             messages = []
#
#             for event in events:
#                 if event['event_type'] == 'message':
#                     if event['agent_id'] is None:
#                         user_message = event['text']
#                         timestamp = event['created_ts']
#                         messages.append({'role': 'user', 'timestamp': timestamp, 'prompt': user_message})
#                     elif event['agent_id'] is not None:
#                         operator_response = event['text']
#                         timestamp = event['created_ts']
#                         messages.append({'role': 'operator', 'timestamp': timestamp, 'prompt': operator_response})
#
#             formatted_chats.append({
#                 "chat_id": chat_id,
#                 "messages": sorted(messages, key=lambda x: x["timestamp"])  # Сортируем по времени
#             })
#
#         formatted_data.append({
#             "client_id": client_id,
#             "chats": formatted_chats
#         })
#
# with open(f'output_{path}', "w", encoding="utf-8") as file:
#     json.dump(formatted_data, file, ensure_ascii=False, indent=4)


# with open(f'output_Jivo_Export_25-02-07_d8706039698e3b7b7e43.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
#
#
# formatted_data = []
#
# for client in data:
#     chats = client['chats']
#     for chat in chats:
#         messages = chat['messages']
#         dialog = []
#
#         user_prompt = []
#         operator_response = []
#         last_role = None
#
#         for msg in messages:
#             role, text = msg['role'], msg['prompt']
#
#             # Если текст пустой — пропускаем этот диалог
#             if text is None:
#                 user_prompt.clear()
#                 operator_response.clear()
#                 last_role = None
#                 continue
#
#             if role == "user":
#                 # Если перед этим был оператор, значит, надо записать прошлый диалог
#                 if last_role == "operator" and operator_response:
#                     dialog.append({
#                         "prompt": " ".join(user_prompt),
#                         "response": " ".join(operator_response)
#                     })
#                     operator_response.clear()
#                     user_prompt.clear()
#
#                 user_prompt.append(text)
#
#             elif role == "operator":
#                 # Если перед этим был пользователь, то начинаем ответ
#                 if last_role == "user" and user_prompt:
#                     operator_response.append(text)
#                 elif last_role == "operator":
#                     operator_response.append(text)
#
#             last_role = role
#
#         if user_prompt and operator_response:
#             dialog.append({
#                 "prompt": " ".join(user_prompt),
#                 "response": " ".join(operator_response)
#             })
#
#         formatted_data.extend(dialog)
#
# with open("last_Jivo_Export_25-02-07_d8706039698e3b7b7e43.json", "w", encoding="utf-8") as file:
#     json.dump(formatted_data, file, indent=4, ensure_ascii=False)
#
# print("Файл успешно обработан! ✅")

# with open('data/last_Jivo_Export_25-02-07_2af1be881764be065221.json', 'r', encoding='utf-8') as file:
#     data1 = json.load(file)
#
# with open('data/last_Jivo_Export_25-02-07_d8706039698e3b7b7e43.json', 'r', encoding='utf-8') as file:
#     data2 = json.load(file)
#
# data = data1 + data2
#
# with open('data/answer-response.json', 'w', encoding='utf-8') as file:
#     json.dump(data, file, indent=4, ensure_ascii=False)


import requests
from bs4 import BeautifulSoup
import re
import json

soup = BeautifulSoup()

main_link = 'https://abit.magtu.ru/'
response = requests.get('https://abit.magtu.ru/?bak&spec').text

specs = BeautifulSoup(response, 'lxml').find('div', class_='json').text
json_data = json.loads(specs.strip())
links = [f'{main_link}/{entry["link"]}' for entry in json_data if "link" in entry]

fields = []

all_profiles_list = list()

def pattern_(text):
    return re.sub(r'\s+', ' ', text).strip()


for link in links:
    response = requests.get(link).text
    with open(f'exceptions/{link.split("/")[-1]}.html', 'w', encoding='utf-8') as file:
        file.write(response)
    napr = BeautifulSoup(response, 'lxml')

    spec_name = napr.find('h1', class_='napravl_title').text
    spec_name = pattern_(spec_name)
    print(spec_name)
    try:
        about = napr.find('section', class_='about').find('p').text
        about = pattern_(about)
    except AttributeError as ex:
        print(ex, spec_name)
        continue
    professions = napr.find('section', class_='prof').find('div').text
    professions = pattern_(professions)

    try:
        profiles = napr.find('div', id='profile').find_all('div', class_='accordion-item')
    except AttributeError as ex:
        print(ex, spec_name)
        continue

    for profile in profiles:

        profile_name = profile.find('h2').text
        profile_name = pattern_(profile_name)

        profile_body = profile.find('div').find('div')

        profile_info = profile_body.find('section', class_='info')

        education_data = dict()

        education_forms = ['Очная форма обучения', 'Очно-заочная форма обучения', 'Заочная форма обучения']

        block_price = [
            pattern_(i.text) for i in profile_info.find('div', class_='block_price').find_all('span', id='price')
                       ]

        block_nums = profile_info.find('div', class_='block_nums').find_all('div', class_='row')[1:]

        for index in range(len(block_nums)):
            row_dict = dict()
            row_dict['Срок обучения'] = pattern_(block_nums[index].find('div', class_='srok').text)
            row_dict['Бюджетные места'] = pattern_(block_nums[index].find('div', class_='budj').text)
            row_dict['Места по договорам'] = pattern_(block_nums[index].find('div', class_='dogov').text)
            row_dict['Стоимость обучения (₽/год)'] = block_price[index]
            education_data[education_forms[index]] = row_dict

        blocks_exams = profile_info.find_all('div', class_='block_exams')
        exams_dict = dict()
        for block in blocks_exams:
            name = pattern_(block.find('div', class_='th isp').text)
            exams = pattern_(block.find('div', class_='vi').text)
            exams_dict[name] = exams

        profile_title = profile.find('p', class_='title') # Чему научат
        profile_skills = ''
        profile_faculty = ''
        try:
            profile_skills = pattern_(profile_title.find_next_sibling('ul').text)
            profile_faculty = pattern_(profile_title.find_next_sibling('p').text)
        except Exception as ex:
            print(ex, spec_name)
        # print(spec_name,
        #       about,
        #       profile_name,
        #       education_data,
        #       exams_dict,
        #       profile_skills,
        #       profile_faculty,
        #       professions,
        #       sep='\n')

        all_profile = {
            'spec_name':spec_name,
            'about': about,
            'profile_name': profile_name,
            'education_data': education_data,
            'exams': exams_dict,
            'profile_skills': profile_skills,
            'profile_faculty': profile_faculty,
            'professions': professions
        }

        all_profiles_list.append(all_profile)

with open('data/specs_magtu.json', 'w', encoding='utf-8') as file:
    json.dump(all_profiles_list, file, ensure_ascii=False, indent=4)
