import os
import PIL

# skill selector
from classifier.Skill_selector import skill_selector

# Models
model_names = []
for root, dirs, files in os.walk("models"):  
    for filename in files:
        model_name = filename.split('.')[0]
        exec(f'from models.{model_name} import {model_name}')
        model_names.append(model_name)

models = {}
for model_name in model_names:
    models[model_name] = eval(f'{model_name}()')

for model_name in models:
    print(model_name)

# Core class
class Core():
    def __init__():
        self.history = []
        self.models = models

    def process_text_ask(self, text):
        model_name = skill_selector.get_predict([text])
        return model_name
    
    def process_photo_ask(self, photo_path):
        pass

core = Core()

for i in range(10):
    core.process_text_ask(input('Введите ваш запрос'))