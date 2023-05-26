import json
import logging
from aiogram import Bot, Dispatcher, executor, types
import os
import matplotlib.pyplot as plt
import time
from PIL import Image
import numpy as np

from config import TG_API_TOKEN
from config import START_TEXT

# skill selector
from Skill_selector_model import skill_selector, context_selector, pc

translator = {
    'gpt': 'GPT_model',
    'dalle': 'DALLE_model',
    'discribe': 'ImageCaption_model'
}

# Models
model_names = []
for root, dirs, files in os.walk("models"):  
    for filename in files:
        model_name = filename.split('.')[0]
        ftype = filename.split('.')[-1]
        if ftype == 'py' and model_name[0] != "_":
            exec(f'from models.{model_name} import {model_name}')
            model_names.append(model_name)

models = {}
for model_name in model_names:
    models[model_name] = eval(f'{model_name}()')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=TG_API_TOKEN)
dp = Dispatcher(bot)

history = {}

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    history[user_id] = []
    
    history[user_id].append(
        {
            'input': message.text,
            'answerer': 'bot',
            'output': START_TEXT
        }
        )
    
    await message.reply(START_TEXT)

answer_processor = {
    'photo': lambda message, filename: bot.send_photo(chat_id=message.chat.id, photo=types.InputFile(filename)),
    'text': lambda message, text: bot.send_message(message.from_user.id, text)  
}
    
@dp.message_handler(content_types=['photo', 'text'])
async def echo(message: types.Message):
    await bot.send_chat_action(message.chat.id, types.ChatActions.TYPING)
    
    inputs = []
    
    if message.photo :
        filename = f'photos/{time.time()}.jpg'
        await message.photo[-1].download(f'C:/Users/Reny/Documents/GitHub/Core/photos/{time.time()}.jpg')
        inputs.append((filename, 'photo'))

    text = ''
    
    if message.text:
        inputs.append((message.text, 'text'))
        text = message.text
        
    if message.caption:
        inputs.append((message.caption, 'text'))
        text = message.caption
        
    if message.photo and message.caption is None:
        return await bot.send_message(message.from_user.id, 'Отправь фотку с текстом, пж')
    
    model_name = translator[skill_selector.get_predict(text)]
    model = models[model_name]
    
    print(text)
    
    answer = model.predict(text, history=history[message.from_user.id])
    
    print(str(zip(answer, model.output_type)))
    
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
    user_id = message.from_user.id
    history[user_id].append(hist)


if __name__ == '__main__':
    try:
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        history['crash_error'] = str(e)
    
    with open("logs.json", "w", encoding="utf-8") as file:
            json.dump(history, file)
            
            