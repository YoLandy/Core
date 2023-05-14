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
        self.model_class, self.tokenizer_class, self.pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.model = self.model_class.from_pretrained(self.pretrained_weights)
        self.model.eval()
        self.device= torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        #init classification model
        self.clf = load(path_to_model) 

    def pad_texts(self, texts_tokenized):
        max_len = 0
        for i in texts_tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in texts_tokenized])
        return padded

    def prepate_data(self, prompts):
        tokenized =  [self.tokenizer.encode(prompts, max_length=512, add_special_tokens=True) for prompt in prompts]
        padded = self.pad_texts(tokenized)
        attentio_mask = np.where(padded != 0, 1, 0)
        return padded, attentio_mask

    def get_embeddings(self, padded, attention_mask):
        output = []
        batch_size = min(32,len(padded))
        cnt = 0
        start = 0

        for idx in tqdm.tnrange(start,len(padded),batch_size):
            batch = torch.tensor(padded[idx:idx+batch_size]).to(self.device)
            local_attention_mask = torch.tensor(attention_mask[idx:idx+batch_size]).to(self.device)

            with torch.no_grad():
                last_hidden_states = self.model(batch,attention_mask = local_attention_mask)[0][:,0,:].cpu().numpy()
                output.append(last_hidden_states)
        features = np.vstack(output)
        return features
    def get_predict(self, prompts):
        padded, attentio_mask = self.prepate_data(prompts)
        features = self.get_embeddings(padded, attentio_mask)
        pred = self.clf.predict(features)
        return(pred)

skill_selector = Skill_selector('lr_clf.joblib')