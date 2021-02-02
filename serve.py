# coding=utf-8
import os
import sys
import time
import numpy as np
from PIL import Image
sys.path.append(os.getcwd())
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import config as static_config 
import torch

config = Cfg.load_config_from_name('vgg_seq2seq')

timeout = float(os.environ.get('timeout','20'))

use_gpu = int(os.getenv("gpu",'0'))

config['cnn']['pretrained']=False

if use_gpu:
    device_name = str(torch.cuda.current_device())
    config['device'] = 'cuda:' + device_name
else:
    config['device'] = 'cpu'

config['predictor']['beamsearch']=False
detector = Predictor(config)

def vietocr_predict():
    def model_api(input_image):
        input_image = Image.fromarray(input_image.astype('uint8'), 'RGB')
        st = time.time()
        s = detector.predict(input_image)
        end = time.time()
        return s,end-st
    return model_api
