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
from Skill_selector_model import skill_selector, context_selector


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
    
    history[user_id].append({
        'input': message.text,
        'answerer': 'bot',
        'output': START_TEXT
    })
    
    await message.reply(START_TEXT)

@dp.message_handler(content_types=['photo'])
async def process_photo(message: types.Message):
    photos = message.photo
    

@dp.message_handler(content_types=['text'])
async def echo(message: types.Message):
    await bot.send_chat_action(message.chat.id, types.ChatActions.TYPING)
    
    text = message.text
    model_name = translator[skill_selector.get_predict(text)]
    model = models[model_name]
    
    if model_name != 'gpt':
        text = context_selector.select_context(message.text, history=history[message.from_user.id])
        print(text)
        
    answer = model.predict(text, history=history[message.from_user.id])
    
    if model.output_type == 'photo':
        filename = f'photos/{time.time()}.png'
        answer.save(filename)
        photo = types.InputFile(filename)
        await bot.send_photo(chat_id=message.chat.id, photo=photo)
        answer = f'photo {filename}'

    if model.output_type == 'text':
        await message.answer(answer)
        
    user_id = message.from_user.id
    history[user_id].append({
        'input': message.text,
        'answerer': model.model_label,
        'output': answer
    })


if __name__ == '__main__':
    try:
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        history['crash_error'] = str(e)
    
    with open("logs.json", "w", encoding="utf-8") as file:
            json.dump(history, file)