import os
import sys
import cv2
import time
import json
import base64
import requests
import argparse
import numpy as np

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

def get_rs(image_path):
    url = "http://localhost:8889/paddle/detect"
    image_base64 = image_to_base64(image_path)
    data = {"img_b64": image_base64}
    result = ""
    try:
        result = requests.put(url=url, data=json.dumps(data), timeout=100).json()
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        raise SystemExit(e)
    return result

def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image    
file_name = "/home/tuannm/studynow/ocr/new_thanos/data/eng_imgs/zXe0q19i5hVbhDYkE15S-1592025728.jpg"
box = get_rs(file_name)
print(box)
image = cv2.imread(file_name)
#box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
image = draw_boxes(image,box)
cv2.imwrite("test.jpg",image)