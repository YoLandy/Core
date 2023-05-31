import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
from sklearn.model_selection import train_test_split
import tqdm
import os
from joblib import dump, load

from models.GPT_model import GPT_model

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

class Task_manager():
    def __init__(self):
        self.gpt = GPT_model()
        self.begin_prompt = "Classification of the problem. Based on the user's request, select only one of the tasks from the list that needs to be solved. "
        self.end_prompt = "Task list = ['conversational', 'text-to-image generation', 'text generation', 'object detection', 'image-to-text', 'image editing', 'image variation', 'image caption', 'control-image-generation', 'image segmentation']. Return only the name of the task in quotation marks, without giving any explanation. If you can't define the task, return the 'conversational'"

    def predict(self, prompt, history):
        prompt = prompt.lower()
        res =  "  User task: '"+ prompt+"'. " + self.end_prompt
        return self.gpt.predict([res], self.begin_prompt, history)[0]

class Skill_selector():
    def __init__(self):
        self.gpt = GPT_model()
        self.begin_prompt = '#2 Model Selection Stage: Given the user request and the parsed tasks, the AI assistant helps the user to select a suitable model from a list of models to process the user request. The assistant should focus more on the description of the model and find the model that has the most potential to solve requests and tasks. Also, prefer models with local inference endpoints for speed and stability'
        self.tm = Task_manager()
        
    def get_model_list(self, model_params=model_params):
        s = 'model list = ['
        for m_name in model_params:
           s += '"'+m_name +'(' + model_params[m_name]['description'] + ')", '
        s+= ']. Do not give me an explanation, just write the name of a suitable model from the list. if you are not sure which model to choose, choose GPT-3.'
        return s

    def predict(self, prompt, history=[], model_params=model_params):
        prompt = prompt.lower()
        end_prompt = self.get_model_list(model_params)
        task = self.tm.predict(prompt,[])
        res =  " parsed_tasks =\""+ prompt+"\"" + end_prompt+ 'User task: \"'+task +'"'
        print(res)
        answer = self.gpt.predict([res], self.begin_prompt, history)[0]

        for m_name in model_params:
            if m_name.lower() in answer.lower(): return m_name
        return 'GPT_model'

skill_selector = Skill_selector()

if __name__ == '__main__':
    print(skill_selector.predict('Draw me a cat'))