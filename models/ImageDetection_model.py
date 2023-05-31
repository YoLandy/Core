from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch
import time


dir_path = ''
cuda = "cuda:1"


class ImageDetection_model():
    def __init__(self):
        
        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        self.image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')

        self.name = 'TinyYolos'
        self.in_types = ['image']  # PIL format
        self.out_types = ['image', 'text']
        self.description = 'find objects on picture, image segmentation, detect instances'
        self.tags = []

    def predict(self, image_path):
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

