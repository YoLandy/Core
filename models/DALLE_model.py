import openai
import numpy as np
import imageio

class DALLE_model():
  API_KEY = '' 

  def __init__(self):
    openai.api_key = self.API_KEY
    self.model = "gpt-3.5-turbo"
    self.imsize = "1024x1024"
    self.history = []
  
  def predict(self, prompt):
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

 
if __name__ == '__main__':
  model = DALLE_model()
  model.render('hello')