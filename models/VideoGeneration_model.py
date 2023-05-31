import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import time

dir_path = ''
cuda = "cuda:1"

class VideoGeneration_model():
    
    def __init__(self):

        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")

        self.model = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", 
                                                       torch_dtype=torch.float16, 
                                                       variant="fp16")
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config)
        #self.model.enable_model_cpu_offload()  -- reduces GPU usage &
        #self.model.enable_vae_slicing()  -- & increases time
        self.model.to(self.device)

        self.name = 'damo-vilab/text-to-video-ms-1.7b'
        self.input_type = ['image', 'text']
        self.output_type = ['image']
        self.description = 'edit a picture by prompt, follow image editing instructions, ' + \
                            'implement requested changes on photo'
        self.tags = []


    def predict(self, prompt):

        video_frames = self.model(prompt, num_inference_steps=30, num_frames=24).frames
        video_path = f'{dir_path}/{time.time()}.mp4'
        
        export_to_video(video_frames, output_video_path=video_path)

        return [video_path]

