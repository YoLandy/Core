from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
import requests
from PIL import Image

dir_path = ''
cuda = "cuda:1"

class ImageQA_model():

    def __init__(self):

        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model.to(self.device)

        
        self.input_type = ['image', 'text']
        self.output_type = ['text']
        self.description = 'visual question answering, image qa'
        self.name = 'vilt-b32-finetuned-vqa' 
        self.tags = []
        

    def predict(self, image_path: str, prompt: str, k=1) -> str:
        # url
        # Image.open(requests.get(url, stream=True).raw)

        image = Image.open(image_path)
        encoding = self.processor(image, prompt, return_tensors="pt")
        encoding.to(self.device)

        outputs = self.model(**encoding)
        logits = outputs.logits
        idxs = torch.topk(logits, k, -1).indices

        softmax = torch.nn.Softmax(dim=-1)
        selected_logits = logits[0][idxs]

        probs = (softmax(selected_logits[0])).tolist()
        preds = [self.model.config.id2label[idxs.tolist()[0][j]] \
                        for j in range(len(probs))]

        if k == 1:
            return preds[0]

        else:
            words = []
            for i in range(len(preds)):
                words.append(f'{preds[i]} with probability {round(probs[i]*100, 1)} %')
            
            return ', '.join(words)

