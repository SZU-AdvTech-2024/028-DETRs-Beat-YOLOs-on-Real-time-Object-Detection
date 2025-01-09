# coding: UTF-8
# @File   : predict.py.py
# @Author : xingxg
# @Date   : 2024/11/28 15:11


from ultralytics import YOLO
from ultralytics import RTDETR


if __name__ == '__main__':

    model = RTDETR("pretrained/rtdetr-x-fgconcat-0.4495.pt")
    model.predict(
        # source="datasets/people_car_dataset/images/val",  # Ctrl + P 参数提示
        source="datasets/test_images",  # Ctrl + P 参数提示
        save=True,
        save_conf=True,
        save_txt=True,
    )  # 设置conf对于结果有影响。
