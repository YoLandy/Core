from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
import time


dir_path = ''
cuda = "cuda:1"


class ImageFromScribble_model():

    def __init__(self):
        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        
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

        self.name = ['sd-controlnet-scribble']
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
        
        filename = f'{dir_path}/{time.time()}.png'
        images.save(filename)
        
        return [filename]