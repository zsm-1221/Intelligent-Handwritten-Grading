"""Hugging Face Spaces entry point for the independently trained grading models."""
import os

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import torch
torch.set_num_threads(max(1,int(os.environ.get('GRADING_THREADS','2'))))
from demo.app import build_app

app=build_app(public_demo=True)

if __name__=='__main__':
    app.queue(default_concurrency_limit=1,max_size=20).launch(
        server_name=os.environ.get('GRADIO_SERVER_NAME','0.0.0.0'),
        server_port=int(os.environ.get('GRADIO_SERVER_PORT','7860')),
        share=False,show_api=False,max_file_size='20mb')
