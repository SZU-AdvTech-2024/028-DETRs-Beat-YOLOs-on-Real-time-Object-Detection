# coding: UTF-8
# @File   : train.py.py
# @Author : xingxg
# @Date   : 2024/11/28 0:36

from ultralytics import RTDETR
from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO("runs/detect/yolo11x_640_50epochs_0.41844/weights/best.pt")
    model = RTDETR("pretrained/rtdetr-fgconcat-fineturn_0.441698.pt")
    model.val(data="datasets/people_car.yaml", batch=1, imgsz=640)  # 设置conf对于结果有影响。


"""
FPS:
=====
yolo11x imgsz: 640
#1 Speed: 0.3ms preprocess, 27.2ms inference, 0.0ms loss, 0.9ms postprocess per image
#2 Speed: 0.3ms preprocess, 27.5ms inference, 0.0ms loss, 0.9ms postprocess per image
#3 Speed: 0.4ms preprocess, 27.5ms inference, 0.0ms loss, 0.9ms postprocess per image
#4 Speed: 0.3ms preprocess, 27.0ms inference, 0.0ms loss, 0.9ms postprocess per image
#5 Speed: 0.3ms preprocess, 27.2ms inference, 0.0ms loss, 0.9ms postprocess per image


yolo11x imgsz: 1280
Speed: 0.6ms preprocess, 85.0ms inference, 0.0ms loss, 1.3ms postprocess per image
Speed: 0.7ms preprocess, 85.0ms inference, 0.0ms loss, 1.5ms postprocess per image


rtdetr-fgconcat



"""