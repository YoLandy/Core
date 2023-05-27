#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import PIL
import requests
import torch
import time
from diffusers import StableDiffusionInstructPix2PixPipeline, \
                      EulerAncestralDiscreteScheduler


dir_path = ''


class ImageEditing_Model():
    def __init__(self):

        self.model =  StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", 
            torch_dtype=torch.float16, 
            safety_checker=None
        )
        self.model.to("cuda")
        self.model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.model.scheduler.config)

        self.name = 'timbrooks/instruct-pix2pix'
        self.input_type = ['image', 'text']
        self.output_type = ['image']
        self.description = 'edit a picture by prompt, follow image editing instructions, \
                            implement requested changes on photo'
        self.tags = []

    def predict(self, image_path, prompt):
        image = PIL.Image.open(image_path)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        images = self.model(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
        
        filename = f'{dir_path}/{time.time()}.png'
        images[0].save(filename)
        
        return [filename]

