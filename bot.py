import json
import logging
from aiogram import Bot, Dispatcher, executor, types
import os
import matplotlib.pyplot as plt
import time
from PIL import Image
import numpy as np

from config import TG_API_TOKEN, START_TEXT, ABSOLUTE_PATH_PHOTO, PHOTO_WITHOUT_CAPTION_ERROR

# skill selector
from OperatingModels import skill_selector, context_selector, pc

translator = {
    'gpt': 'GPT_model', 
    'dalle': 'DALLE_model',
    'discribe': 'ImageCaption_model'
}

# Models import
model_names = []
for root, dirs, files in os.walk("models"):  
    for filename in files:
        model_name = filename.split('.')[0]
        ftype = filename.split('.')[-1]
        if ftype == 'py' and model_name[0] != "_":
            exec(f'from models.{model_name} import {model_name}')
            model_names.append(model_name)

# Models inits
models = {}
for model_name in model_names:
    models[model_name] = eval(f'{model_name}()')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=TG_API_TOKEN)
dp = Dispatcher(bot)

history = {}

# start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    history[user_id] = []
    
    history[user_id].append({
        'input': message.text,
        'answerer': 'bot',
        'output': START_TEXT
    })
    
    await message.reply(START_TEXT)

# функции которые обрабатывают разные типы данных
answer_processor = {
    'photo': lambda message, filename: bot.send_photo(chat_id=message.chat.id, photo=types.InputFile(filename)),
    'text': lambda message, text: bot.send_message(message.from_user.id, text)  
}
    
@dp.message_handler(content_types=['photo', 'text'])
async def echo(message: types.Message):
    user_id = message.from_user.id
    inputs = []
    text = ''
    
    # если нет истории, то создаем
    if user_id not in history:
        history[user_id] = []
    
    # если есть текст в сообщении - сохраняем message.text
    if message.text:
        inputs.append((message.text, 'text'))
        text = message.text
    
    # если есть описание к фотке - сохраняем message.caption
    if message.caption:
        inputs.append((message.caption, 'text'))
        text = message.caption

    # если есть фотка без описания
    if message.photo and message.caption is None:
        return await bot.send_message(message.from_user.id, PHOTO_WITHOUT_CAPTION_ERROR)
    
    model_name = translator[skill_selector.get_predict(text)]
    model = models[model_name]
    
    #пока реализованы только нейросети которые принимают одну фотку
    if message.photo :
        filename = f'{ABSOLUTE_PATH_PHOTO}/{time.time()}.jpg'
        await message.photo[-1].download(f'{ABSOLUTE_PATH_PHOTO}/{time.time()}.jpg')
        inputs.append((filename, 'photo'))
        answer = model.predict(filename, history=history[message.from_user.id])
    
    else:
        answer = model.predict(text, history=history[message.from_user.id])
    
    # отправляем все что отправила нейросеть
    for value, value_type in zip(answer, model.output_type):
        await answer_processor[value_type](message, value)
    
    # создаем историю
    hist = {
        'input': inputs,
        'answerer': model.model_label,
        'output': list(zip(answer, model.output_type))
    }
    
    hist = pc.get_text_from_history(hist)
    
    # сохраняем
    history[user_id].append(hist)


if __name__ == '__main__':
    try:
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        history['crash_error'] = str(e)
    
    with open("logs.json", "w", encoding="utf-8") as file:
            json.dump(history, file)