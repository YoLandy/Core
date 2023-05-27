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
from models.ImageCaption_model import ImageCaption_model

class Skill_selector:
    def __init__(self):
        discriptions = {'\'GPT-3\'': '',
               '\'DALLE\'': 'draw picture by text',
               '\'Discriber\'': 'discribe image and photo, image caption model'}
        
        self.gpt = GPT_model()
        self.m_names = ""
        self.begin_prompt =  "Imagine that you have a choice of "
        self.middle_prompt = "You need to answer the request: "
        self.end_prompt = " .Which model will you choose to solve this problem? in the answer, output only the name of one model!"
        for m_name in discriptions:
            self.m_names+=m_name+ ' ('+ discriptions[m_name] +') '
        
    def get_predict(self, prompt):
        prompt = prompt.lower()
        res = self.begin_prompt + self.m_names + self.middle_prompt+"\""+ prompt+"\"" + self.end_prompt
        return self.check_result(self.gpt.predict(res)[0])
    
    def check_result(self,res):
        if 'dalle' in res.lower(): return 'dalle'
        else:
            if 'discriber' in res.lower(): return 'discribe'
            else: return 'gpt'

skill_selector = Skill_selector()

class Context_selector():
    def __init__(self):
        self.gpt = GPT_model()
        self.begin_promt = "which element of the text does the message refer to: "
        self.end_prompt = " in the response, output this full text element. If there is no such element, output \"None\"."

    def select_context(self, prompt, history):
        res = self.begin_promt + prompt + self.end_prompt
        tmp = self.gpt.predict(res, history)
        if 'none' in tmp.lower(): return prompt
        else: return tmp
        
    
context_selector = Context_selector()

class Photo_convertor():
    def __init__(self):
        self.convertor = ImageCaption_model()

    def get_text_from_history(self, history):
        for value, value_type in history['input']:
            if value_type == 'photo':
                history['input_disc']  = 'the photo contains "' +self.convertor.predict(value)[0] + '"'
        for value, value_type in history['output']:
            if value_type == 'photo':
                history['output_disc']  = 'the photo contains "' +self.convertor.predict(value)[0] + '"'    
        
        return history
pc = Photo_convertor()