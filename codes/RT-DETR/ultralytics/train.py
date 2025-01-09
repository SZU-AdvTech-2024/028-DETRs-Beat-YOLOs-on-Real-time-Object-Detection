# coding: UTF-8
# @File   : train.py.py
# @Author : xingxg
# @Date   : 2024/11/28 0:36

from ultralytics import RTDETR
from ultralytics import YOLO

if __name__ == '__main__':

    # model = RTDETR("datasets/rtdetr-l.yaml").load("pretrained/rtdetr-l.pt")
    # model.train(
    #     data="datasets/people_car.yaml",
    #     epochs=100,
    # )



    model = YOLO("datasets/yolo11.yaml").load("yolo11n.pt")
    model.train(
        data="datasets/people_car.yaml",
        epochs=100,
        batch=24,
    )





