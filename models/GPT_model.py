import openai
import numpy as np
import imageio
from config import GPT_API_KEY

from abstract_model import OmniModel

class GPT_model(OmniModel):
  def __init__(self):
    openai.api_key = GPT_API_KEY
    super().__init__()
    self.model = 'gpt-3.5-turbo'
    self.input_type = 'text'
    self.output_type = 'text'
    self.discription = 'language_model'
    self.model_label = 'gpt'

  def predict(self, message, history=[]):
    response = openai.ChatCompletion.create(
      model=self.model,
      messages=self.get_messages(history, message)
    )
    answer = response['choices'][0]['message']['content']

    return answer

  def get_messages(self, history, message):
    messages = []
    for hist in history:
      if hist['answerer'] == self.model_label:
        messages.append({'role': "user", 'content': hist['input']})
        messages.append({'role': "assistant", 'content': hist['output']})
    messages.append({"role": "user", "content": message})
    return messages

  
if __name__ == '__main__':
  model = GPT_model()
  print(model.render('hello'))