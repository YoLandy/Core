import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog as fd  

# skill selector
from classifier.Skill_selector import skill_selector

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

# Core class
class Core():
    def __init__(self):
        self.history = []
        self.models = models

    def process_text_ask(self, text):
        print(skill_selector.get_predict([text]))
        model_name = translator[skill_selector.get_predict([text])[0]]
        model = self.models[model_name] 
        return model.predict(text)

    def process_photo_ask(self, photo_path):
        pass

core = Core()
    
while True:
    input_text = input('Введите ваш запрос: \n')
    
    if input_text[0] == '$':
        command = input()
        if command == 'photo':
            photo_file_name = fd.askopenfilename()
            print(photo_file_name)
        if command == 'exit':
            break
        if command == 'help':
            print('photo: load photo, exit: to exit')
    else:
        output = core.process_text_ask(input_text)
        if type(output) == str:
            print(output)
        
        if type(output) == np.ndarray:
            plt.imshow(output)
            plt.show()