from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch
import time

import config

dir_path = config.ABSOLUTE_PATH_PHOTO
cuda = config.CUDA

class ImageDetection_model():
    def __init__(self):
        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
        self.image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-small')
        self.model.to(self.device)

        self.name = 'TinyYolos'
        self.input_type = ['image']  # PIL format
        self.output_type = ['image', 'text']
        self.description = 'find objects on picture, image segmentation, detect instances.'
        self.tags = []
        self.model_label = 'image detection'

    def predict(self, inputs, history=[]):
        image_path = inputs[0]
        image = Image.open(image_path)
        
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        inputs = self.image_processor(images=image, return_tensors='pt')
        outputs = self.model(**inputs)

        logits = outputs.logits
        bboxes = outputs.pred_boxes

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, \
                                       threshold=0.9, target_sizes=target_sizes)[0]

        text = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            text += (f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}") + '\n'
            
            draw = ImageDraw.Draw(image)
            draw.rectangle(box, width=3, outline='yellow')
            
        filename = f'{dir_path}/{time.time()}.png'
        image.save(filename)

        return [filename, text]

