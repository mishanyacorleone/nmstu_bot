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


# import requests
# from bs4 import BeautifulSoup
# import re
# import json
#
# soup = BeautifulSoup()
#
# main_link = 'https://abit.magtu.ru/'
# response = requests.get('https://abit.magtu.ru/?bak&spec').text
#
# specs = BeautifulSoup(response, 'lxml').find('div', class_='json').text
# json_data = json.loads(specs.strip())
# links = [f'{main_link}/{entry["link"]}' for entry in json_data if "link" in entry]
#
# fields = []
#
# all_profiles_list = list()
#
# def pattern_(text):
#     return re.sub(r'\s+', ' ', text).strip()
#
#
# for link in links:
#     response = requests.get(link).text
#     with open(f'exceptions/{link.split("/")[-1]}.html', 'w', encoding='utf-8') as file:
#         file.write(response)
#     napr = BeautifulSoup(response, 'lxml')
#
#     spec_name = napr.find('h1', class_='napravl_title').text
#     spec_name = pattern_(spec_name)
#     print(spec_name)
#     try:
#         about = napr.find('section', class_='about').find('p').text
#         about = pattern_(about)
#     except AttributeError as ex:
#         print(ex, spec_name)
#         continue
#     professions = napr.find('section', class_='prof').find('div').text
#     professions = pattern_(professions)
#
#     try:
#         profiles = napr.find('div', id='profile').find_all('div', class_='accordion-item')
#     except AttributeError as ex:
#         print(ex, spec_name)
#         continue
#
#     for profile in profiles:
#
#         profile_name = profile.find('h2').text
#         profile_name = pattern_(profile_name)
#
#         profile_body = profile.find('div').find('div')
#
#         profile_info = profile_body.find('section', class_='info')
#
#         education_data = dict()
#
#         education_forms = ['Очная форма обучения', 'Очно-заочная форма обучения', 'Заочная форма обучения']
#
#         block_price = [
#             pattern_(i.text) for i in profile_info.find('div', class_='block_price').find_all('span', id='price')
#                        ]
#
#         block_nums = profile_info.find('div', class_='block_nums').find_all('div', class_='row')[1:]
#
#         for index in range(len(block_nums)):
#             row_dict = dict()
#             row_dict['Срок обучения'] = pattern_(block_nums[index].find('div', class_='srok').text)
#             row_dict['Бюджетные места'] = pattern_(block_nums[index].find('div', class_='budj').text)
#             row_dict['Места по договорам'] = pattern_(block_nums[index].find('div', class_='dogov').text)
#             row_dict['Стоимость обучения (₽/год)'] = block_price[index]
#             education_data[education_forms[index]] = row_dict
#
#         blocks_exams = profile_info.find_all('div', class_='block_exams')
#         exams_dict = dict()
#         for block in blocks_exams:
#             name = pattern_(block.find('div', class_='th isp').text)
#             exams = pattern_(block.find('div', class_='vi').text)
#             exams_dict[name] = exams
#
#         profile_title = profile.find('p', class_='title') # Чему научат
#         profile_skills = ''
#         profile_faculty = ''
#         try:
#             profile_skills = pattern_(profile_title.find_next_sibling('ul').text)
#             profile_faculty = pattern_(profile_title.find_next_sibling('p').text)
#         except Exception as ex:
#             print(ex, spec_name)
#         # print(spec_name,
#         #       about,
#         #       profile_name,
#         #       education_data,
#         #       exams_dict,
#         #       profile_skills,
#         #       profile_faculty,
#         #       professions,
#         #       sep='\n')
#
#         all_profile = {
#             'spec_name':spec_name,
#             'about': about,
#             'profile_name': profile_name,
#             'education_data': education_data,
#             'exams': exams_dict,
#             'profile_skills': profile_skills,
#             'profile_faculty': profile_faculty,
#             'professions': professions
#         }
#
#         all_profiles_list.append(all_profile)
#
# with open('data/specs_magtu.json', 'w', encoding='utf-8') as file:
#     json.dump(all_profiles_list, file, ensure_ascii=False, indent=4)

'''Парсинг приложение №2'''


# import requests
# from bs4 import BeautifulSoup
# import re
# import json
#
# url = 'https://www.magtu.ru/abit/17769-perechen-vi-soo-vo-2025-bs.html'
#
# response = requests.get(url).text
#
# soup = BeautifulSoup(response, 'lxml')
#
# table = soup.find('div', class_='table_wrapper').find('table').find('tbody').find_all('tr')[2:]
#
# vi = list()
#
# for string in table:
#     spec = string.find_all('td')
#
#     code_spec = spec[0].text # Код специальности
#     name_spec = spec[1].text # Название специальности
#     vi_1 = re.sub(r'\d+', '', spec[2].text).replace('математика', 'профильная математика', 1) # Первое вступительное испытание
#     vi_2 = re.sub(r'\d+', '', spec[3].text).replace('математика', 'профильная математика', 1) # Второе вступительное испытание
#     vi_3 = re.sub(r'\d+', '', spec[4].text).replace('математика', 'профильная математика', 1) # Третье вступительное испытание
#     if vi_3.count('/'):
#         vi_3 = vi_3.replace('/', 'ИЛИ')
#     vi_4 = re.sub(r'\d+', '', spec[5].text).replace('математика', 'профильная математика', 1) # Четвертое вступительное испытание
#     if len(vi_4) <= 1:
#         vi_4 = 'Нет'
#
#     vi.append({
#         'Код специальности': code_spec,
#         'Направление и профиль подготовки': name_spec,
#         'Вступительное испытание №1': vi_1,
#         'Вступительное испытание №2': vi_2,
#         'Вступительное испытание №3': vi_3,
#         'Вступительное испытание №4': vi_4
#     })
#
# with open('Приложение №2 (Вступительные испытания ШКОЛА).json', 'w', encoding='utf-8') as f:
#     json.dump(vi, f, ensure_ascii=False, indent=2)


'''Парсинг приложение №3'''
# import requests
# from bs4 import BeautifulSoup
# import re
# import json
#
# url = 'https://www.magtu.ru/abit/17773-perechen-vi-spo-2025-bs.html'
#
# response = requests.get(url).text
#
# soup = BeautifulSoup(response, 'lxml')
#
# table = soup.find('div', class_='table_wrapper').find('table').find('tbody').find_all('tr')[2:]
#
# vi = list()
#
# for string in table:
#     spec = string.find_all('td')
#
#     code_spec = spec[0].text # Код специальности
#     name_spec = spec[1].text # Название специальности
#     vi_1 = re.sub(r'\d+', '', spec[2].text) # Первое вступительное испытание
#     vi_2 = re.sub(r'\d+', '', spec[3].text) # Второе вступительное испытание
#     vi_3 = re.sub(r'\d+', '', spec[4].text) # Третье вступительное испытание
#     if vi_3.count('/'):
#         vi_3 = vi_3.replace('/', 'ИЛИ')
#     vi_4 = re.sub(r'\d+', '', spec[5].text) # Четвертое вступительное испытание
#     if len(vi_4) <= 1:
#         vi_4 = 'Нет'
#
#     vi.append({
#         'Код специальности': code_spec,
#         'Направление и профиль подготовки': name_spec,
#         'Вступительное испытание №1': vi_1,
#         'Вступительное испытание №2': vi_2,
#         'Вступительное испытание №3': vi_3,
#         'Вступительное испытание №4': vi_4
#     })
#
# with open('Приложение №3 (Вступительные испытания СПО).json', 'w', encoding='utf-8') as f:
#     json.dump(vi, f, ensure_ascii=False, indent=2)


'''Парсинг приложение №5'''
# import requests
# from bs4 import BeautifulSoup
# import re
# import json
#
#
# url = 'http://magtu.ru/515-abiturientam/informirovanie-o-prijome-2025/17775-min-max-bally-2025-bs.html'
#
# response = requests.get(url).text
# soup = BeautifulSoup(response, 'lxml')
#
# table = soup.find('div', class_='table_wrapper').find('table').find('tbody').find_all('tr')[3:]
#
# vi = list()
#
# c = 0
# for tr in table:
#     c += 1
#     if c == 41:
#         break
#
#     if c != 12 and c != 25 and c != 41:
#         vi_s = tr.find_all('td')
#         vi_name = vi_s[0].text.lower().replace('математика', 'профильная математика')
#         print(vi_name)
#         min_mark = vi_s[1].text
#         max_mark = vi_s[2].text
#         form_conduct = vi_s[3].text
#         vi_lang = vi_s[4].text
#         venue_vi = vi_s[5].text
#
#         vi.append({
#             'Наименование вступительного испытания (ВИ)': vi_name,
#             'Минимальное количество баллов': min_mark,
#             'Максимальное количество баллов': max_mark,
#             'Форма проведения': form_conduct,
#             'Язык, на котором осуществляется сдача ВИ': vi_lang,
#             'Место проведения ВИ': venue_vi
#         })
#
# with open('Приложение №5 (Минимальное и максимальное количество баллов).json', 'w', encoding='utf-8') as f:
#     json.dump(vi, f, ensure_ascii=False, indent=2)

'''Объединение приложения №5 и Приложений №2 и №3'''

# import json
# import re
#
# # Загрузка данных
# with open("Приложения итог/temp/Приложение №3 (Вступительные испытания СПО).json", "r", encoding="utf-8") as f:
#     specs = json.load(f)
#
# with open("Приложения итог/Приложение №5 (Минимальное и максимальное количество баллов).json", "r", encoding="utf-8") as f:
#     exams_info = json.load(f)
#
# # Индексируем Приложение 5 по названию экзамена
# exam_scores = {}
# for exam in exams_info:
#     name = exam["Наименование вступительного испытания (ВИ)"].lower()
#     exam_scores[name] = {
#         "min": exam["Минимальное количество баллов"],
#         "max": exam["Максимальное количество баллов"]
#     }
#
# # Обработка каждой специальности
# for spec in specs:
#     scores_summary = []
#
#     # Собираем все ключи с ВИ
#     for i in range(1, 5):
#         key = f"Вступительное испытание №{i}"
#         if key in spec and spec[key].lower() != "нет":
#             subjects = re.split(r"ИЛИ|или", spec[key])
#             for subj in subjects:
#                 subj_clean = subj.strip().lower()
#                 if subj_clean in exam_scores:
#                     score_data = exam_scores[subj_clean]
#                     scores_summary.append(
#                         f"{subj.strip()} — минимально {score_data['min']} баллов, максимально {score_data['max']} баллов"
#                     )
#                 else:
#                     scores_summary.append(f"{subj.strip()} — данные не найдены")
#
#     # Добавляем итоговое поле
#     spec["Минимальное и максимальное количество баллов"] = "; ".join(scores_summary)
#
# # Сохраняем результат
# with open("Приложения итог/Приложение №3 новое.json", "w", encoding="utf-8") as f:
#     json.dump(specs, f, ensure_ascii=False, indent=2)

'''Приложение №9'''

# import requests
# from bs4 import BeautifulSoup
# import re
# import json
#
#
# url = 'http://magtu.ru/515-abiturientam/informirovanie-o-prijome-2025/17779-perechen-indiv-dostizh-2025-bs.html'
#
# response = requests.get(url).text
# soup = BeautifulSoup(response, 'lxml')
#
# table = soup.find('div', class_='table_wrapper').find('table').find('tbody').find_all('tr')[1:]
#
# ind_dost = list()
# c = 0
# for tr in table:
#     c += 1
#     if c == 14:
#         continue
#     td_s = tr.find_all('td')
#     print(td_s)
#     name_id = td_s[0].text
#     docs = td_s[1].text
#     mark = td_s[2].text
#     ind_dost.append({
#         'Наименование индивидуального достижения': name_id,
#         'Документы, подтверждающие получение индивидуального достижения': docs,
#         'Балл': mark
#     })
#
# with open("Приложения итог/Приложение №9.json", "w", encoding="utf-8") as f:
#     json.dump(ind_dost, f, ensure_ascii=False, indent=2)


'''Проходные баллы'''
# import requests
# from bs4 import BeautifulSoup
# import re
# import json
#
#
# url = 'https://www.magtu.ru/abit/points.php'
#
# response = requests.get(url).text
# soup = BeautifulSoup(response, 'lxml')
#
# points_2024_trs = soup.find('div', id='2024').find('table', class_='points').find('tbody').find_all('tr')
# points_2023_trs = soup.find('div', id='2023').find('table', class_='points').find('tbody').find_all('tr')
#
# result_points = {}
# print(len(points_2023_trs), len(points_2024_trs))
#
# for tr in points_2023_trs:
#     spec_tds = tr.find_all('td')
#     name_spec = spec_tds[0].text.strip().replace('    ', ' ')
#     point = spec_tds[1].text.strip().replace('    ', ' ')
#     result_points[name_spec] = f'2023 год - {point}'
#
# for tr in points_2024_trs:
#     spec_tds = tr.find_all('td')
#     name_spec = spec_tds[0].text.strip()
#     point = spec_tds[1].text.strip()
#     result_points[name_spec] = f'2024 год - {point}'
#
# with open('Приложения итог/Проходные баллы 2023-2024 год.json', 'w', encoding='utf-8') as file:
#     json.dump(result_points, file, ensure_ascii=False, indent=2)


import requests

response = requests.get('https://www.gosuslugi.ru/vuznavigator/specialties/1182').text
with open('response.html', 'w', encoding='utf-8') as f:
    f.write(str(response))
print(response)