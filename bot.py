from aiogram import Bot, Dispatcher, executor, types

import json
import logging
import time

from config import TG_API_TOKEN, START_TEXT, ABSOLUTE_PATH_PHOTO, ABSOLUTE_PATH_AUDIO, ABSOLUTE_PATH_VIDEO, PHOTO_WITHOUT_CAPTION_ERROR, ABSOLUTE_PATH_LOGS

# подгрузим модельки
from ModelCollector import models

# skill selector и остальные
from OperatingModels import skill_selector

model_params = {
  'GPT_model': {
    'description': '"text-to-text", "text generation", "conversational"',
    'inputs': ['text'],
    'outputs': ['image'],
  },
  'DALLE_model': {
    'description': '"text-to-image", "text-to-image generation", draws and paints what is written in the text, create an image by request, picture generation',
    'inputs': ['text'],
    'outputs': ['image'],
  },
  'ImageCaption_model' : {
    'description': ' "image-to-text", "image caption", describe what is happening on a picture, image captioning',
    'inputs': ['image'],
    'outputs': ['text'],
  },
  'ImageDetection_model': {
    'description': ' "image segmentation", "object detection", find objects on picture, image segmentation, detect instances',
    'inputs': ['image'],
    'outputs': ['image', 'text']
  },
  'ImageEditing_model': {
      'description': ' "image-to-image", "image editing" edit a picture by prompt, follow image editing instructions, implement requested changes on photo',
      'inputs': ['image', 'text'],
      'outputs': ['image'],
  },
  'ImageFromScribble_model': {
      'description': '"image-to-image", "control-image-generation", "image variation", completes the image',
      'inputs': ['image', 'text'],
      'outputs': ['image'],
  }
}


# вот тут бред надо исправить
translator = {
    'gpt': 'GPT_model', 
    'dalle': 'DALLE_model',
    'discribe': 'ImageCaption_model'
}

# по идее в скилл селектор надо передать дескриптионы
# skill_selector.descriptions = model_descriptions

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=TG_API_TOKEN)
dp = Dispatcher(bot)

# функции которые обрабатывают разные типы данных
answer_processor = {
    'image': lambda message, filename: bot.send_photo(chat_id=message.chat.id, photo=types.InputFile(filename)),
    'text': lambda message, text: bot.send_message(message.from_user.id, text)  
}

#история для gpt
history = {}


# start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    history[user_id] = []
    
    history[user_id].append({
        'input': (message.text, 'text'),
        'answerer': 'bot',
        'output': START_TEXT
    })    
    await message.reply(START_TEXT)

@dp.message_handler()
async def echo(message: types.Message):
    user_id = message.from_user.id
    hist_inputs = []
    
    # если нет истории, то создаем
    if user_id not in history:
        history[user_id] = []
        history[user_id].append({
            'input': (message.text, 'text'),
            'answerer': 'bot',
            'output': START_TEXT
        })
    
    # если есть фотка без описания
    if message.photo and not message.caption:
        return await bot.send_message(message.from_user.id, PHOTO_WITHOUT_CAPTION_ERROR)
    
    if message.text:
        text = message.text
    else:
        text = message.caption
    
    hist_inputs.append((text, 'text'))
    
    #пока реализованы только нейросети которые принимают одну фотку
    if message.photo:
        filename = f'{ABSOLUTE_PATH_PHOTO}/{time.time()}.jpg'
        await message.photo[-1].download(f'{ABSOLUTE_PATH_PHOTO}/{time.time()}.jpg')
        hist_inputs.append((filename, 'image'))
    
    model_name = skill_selector.predict(text, model_params=model_params, history=[])
    print()
    print(text)
    print(model_name)
    model = models[model_name]
    inputs = []
    for data_type in model.input_type:
        if data_type == 'image':
            inputs.append(filename)
        if data_type == 'text':
            inputs.append(text)
    
    answer = model.predict(inputs, history=history[message.from_user.id])

    # отправляем все что отправила нейросеть
    for value, value_type in zip(answer, model.output_type):
        await answer_processor[value_type](message, value)
    
    # создаем историю
    hist = {
        'input': hist_inputs,
        'answerer': model.model_label,
        'output': list(zip(answer, model.output_type))
    }
    
    # сохраняем
    history[user_id].append(hist)


if __name__ == '__main__':
    try:
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        history['crash_error'] = str(e)
    
    with open(f"{ABSOLUTE_PATH_LOGS}/logs{time.time()}.json", "w", encoding="utf-8") as file:
            json.dump(history, file)