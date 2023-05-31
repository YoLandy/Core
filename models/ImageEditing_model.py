import PIL
import requests
import torch
import time
from diffusers import StableDiffusionInstructPix2PixPipeline, \
                      EulerAncestralDiscreteScheduler

import config

dir_path = config.ABSOLUTE_PATH_PHOTO
cuda = config.CUDA


class ImageEditing_model():
    def __init__(self):

        self.model =  StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", 
            torch_dtype=torch.float16, 
            safety_checker=None
        )
        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.model.scheduler.config)
        self.model.to(self.device)

        self.name = 'instruct-pix2pix'
        self.input_type = ['image', 'text']
        self.output_type = ['image']
        self.description = 'edit a picture by prompt, follow image editing instructions, ' + \
                            'implement requested changes on photo'
        self.tags = []
        self.model_label = 'image editing'

    def predict(self, inputs, history=[]):
        image_path, prompt = inputs
        image = PIL.Image.open(image_path)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        images = self.model(prompt, image=image, num_inference_steps=9, image_guidance_scale=2).images
        
        filename = f'{dir_path}/{time.time()}.png'
        images[0].save(filename)
        
        return [filename]

