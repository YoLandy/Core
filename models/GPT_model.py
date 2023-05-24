import openai
import numpy as np
import imageio

class GPT_model():
  API_KEY = '' 

  def __init__(self):
    openai.api_key = self.API_KEY
    self.model = "gpt-3.5-turbo"
    self.imsize = "1024x1024"
    self.history = []

  def predict(self, message):
    self.history.append({
        'data' : {"role": "user", "content": message},
        'type': 'text_ask'
    })
    
    response = openai.ChatCompletion.create(
      model=self.model,
      messages=self.get_messages()
    )
    answer = response['choices'][0]['message']['content']
    
    self.history.append({
        'data' : {"role": "assistant", "content": answer},
        'type': 'text_ans'
    })

    return answer

  def get_messages(self):
    return [ask['data'] for ask in self.history if ask['type'] in ['text_ask', 'text_ans']]
  
if __name__ == '__main__':
  model = GPT_model()
  print(model.render('hello'))