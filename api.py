# coding=utf-8
import os
import shutil
import sys
import time
import base64
import traceback
import logging
import json
import cv2
import numpy as np
#from flask import Flask,jsonify,request
from PIL import Image
from io import BytesIO
#from gevent.pywsgi import WSGIServer

from typing import Optional
from fastapi import FastAPI

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

app = FastAPI()

sys.path.append(os.getcwd())

logger = logging.getLogger('tharact_api')

c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('static/tharact.log')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.ERROR)

c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)

SERVICE_NAME = "paddle_det"

from pydantic import BaseModel
class Item(BaseModel):
    img_b64: str

@app.get("/")
def ping():
    return jsonify({'paddle_det_API': {'Status': 'OK'}}), 200

@app.put("/paddle/detect")
def uet(item: Item):
    data = {}
    det_time = 0
    try:
        #dataDict = json.loads(request.data.decode('utf-8'))
        
        try:
            img_b64 = item.img_b64
            #text_type = dataDict.get("text_type", "printed")
                
            try:
                        
                img_data = Image.open(BytesIO(base64.b64decode(img_b64))).convert('RGB')
                img_data = np.array(img_data)
                imgs = img_data[:, :, :: 1]

                texts = ""
                st = time.time()
                result = ocr.ocr(imgs, rec=False)
                end = time.time()
                det_time = end-st
                data['result'] = len(result)
                
            except Exception as e:
                logger.error({"error":"fail to detect","tuannm10": "paddle", "stacktrace": str(traceback.format_exception(None, e, e.__traceback__))})
                return data,204
        except Exception as e: 
            logger.error({"error": "fail to load json dict input", "tuannm10": "paddle", "stacktrace": str(traceback.format_exception(None, e, e.__traceback__))})
            return data,406
    except Exception as e:
        logger.error({"tuannm10": "paddle", "stacktrace": str(traceback.format_exception(None, e, e.__traceback__))})
        return data,400 
    logger.info("200 ok request ,paddle time : {}".format(det_time))
    return result


if __name__ == '__main__':
    app.run(host="0.0.0.0",port="8889")
