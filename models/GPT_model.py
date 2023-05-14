import openai
import numpy as np
import imageio

class GPT_model():
  API_KEY = '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' 

  def __init__(self):
    openai.api_key = self.API_KEY
    self.model = "gpt-3.5-turbo"
    self.imsize = "1024x1024"
    self.history = []

  def ask(self, message):
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
  
  def draw(self, prompt):
    self.history.append({
        'data': prompt,
        'type': 'picture_prompt',
    })

    response = openai.Image.create(
      prompt=prompt,
      n=1,
      size=self.imsize
    )
    image_url = response['data'][0]['url']
    image = self.url_to_numpy(image_url)
    
    self.history.append({
        'data': image,
        'type': 'picture'
    })

    return image

  def url_to_numpy(self, url):
    image = imageio.imread(url)
    return np.asarray(image)

  def get_messages(self):
    return [ask['data'] for ask in self.history if ask['type'] in ['text_ask', 'text_ans']]