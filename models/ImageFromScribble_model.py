# -*- coding: utf-8 -*-
"""ImageFromScribble_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MNB2vYw2wef6OV79VCRdOf_DyvUS1B2J
"""

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector

dir_path = ''

class ImageFromScribble_model():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        self.model = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", 
            torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=self.model,
            torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.to(self.device)
        #self.model.to(self.device)

        self.input_type = ['image', 'text']
        self.output_type = ['image']
        self.description = 'create similar pictures, variations from following prompt, ' + \
                            'make images that look like the original,' + \
                            'image from scratch, scribble'
        self.tags = []


    def predict(self, image_path, text):
        image = Image.open(image_path)
        image = self.hed(image, scribble=True)

        images = self.pipe(text, image, num_inference_steps=20).images[0]
        #image.save(f'{dir_path}/{time.time()}.png')

        return images