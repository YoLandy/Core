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


class Skill_selector():
    def __init__(self, path_to_model):
        #init bert
        self.model_class,  self.tokenizer_class,  self.pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.tokenizer =  self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.model =  self.model_class.from_pretrained(self.pretrained_weights)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.eval()
        self.model.to(self.device)
        #init pred_model
        self.clf = load(path_to_model) 
        
    def pad_texts(self, texts_tokenized):
        max_len = 0
        for i in texts_tokenized:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0]*(max_len-len(i)) for i in texts_tokenized])
        return padded

    def attention_mask(self, padded_text):
        return np.where(padded_text != 0, 1, 0)

    def get_embedings(self, prompts):
        tokenized =  [self.tokenizer.encode(x_idx, max_length=512,truncation=True, add_special_tokens=True) for x_idx in prompts]
        padded = self.pad_texts(tokenized)
        attention_mask_train = self.attention_mask(padded)

        output = []
        batch_size = 32
        for idx in range(0,len(padded),batch_size):
            batch = torch.tensor(padded[idx:idx+batch_size]).to(self.device)
            local_attention_mask = torch.tensor(attention_mask_train[idx:idx+batch_size]).to(self.device)
            with torch.no_grad():
                last_hidden_states = self.model(batch,attention_mask = local_attention_mask)[0][:,0,:].cpu().numpy()
                output.append(last_hidden_states)
        embeddings = np.vstack(output)
        return embeddings
    
    def get_predict_by_emb(self, emb):
        pred = self.clf.predict(emb)       
    
    def get_predict(self, prompts):
        emb = self.get_embedings(prompts)
        pred = self.clf.predict(emb)
        return pred
    
    def get_score(self, prompts, answers):
        emb = self.get_embedings(prompts)
        return self.clf.score(emb, answers)

skill_selector = Skill_selector('C:/Users/Reny/Documents/GitHub/Core/classifier/preset_models/multy_label_clf.joblib')