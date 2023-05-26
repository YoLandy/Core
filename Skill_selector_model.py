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

class Skill_selector:
    def __init__(self):
        discriptions = {'\'GPT-3\'': '',
               '\'DALLE\'': ''}
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
        return self.check_result(self.gpt.predict(res))
    
    def check_result(self,res):
        if 'dalle' in res.lower(): return 'dalle'
        if 'gpt' in res.lower(): return 'gpt'

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