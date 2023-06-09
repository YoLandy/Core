from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import time

import config

cuda = config.CUDA


class ImageCaption_model():
    def __init__(self):
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
        

    def predict(self, prompt: dict) -> dict:
        image = Image.open(prompt['image'])
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        pixel_values = self.feature_extractor(images=image, 
                                              return_tensors="pt").pixel_values
        pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        
        return {'text': preds[0]}