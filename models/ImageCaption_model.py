# -*- coding: utf-8 -*-
"""ImageCaptionModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jwIFuE2DymkA_OKh6Plw-fuewHygE5GH
"""

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import time


dir_path = ''
cuda = "cuda:0"


class ImageCaption_model():
    def __init__(self):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        
        self.input_type = ['image']  # PIL
        self.output_type = ['text']
        self.description = 'describe what is happening on a picture, image captioning'
        self.name = 'image-caption' 
        self.tags = []
        

    def predict(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        pixel_values = self.feature_extractor(images=image, 
                                              return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        
        filename = f'{dir_path}/{time.time()}.png'
        preds[0].save(filename)
        
        return [filename]