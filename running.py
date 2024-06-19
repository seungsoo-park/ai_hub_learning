from ultralytics import YOLO
import os


if __name__ == '__main__':
    model = YOLO("yolov8x.pt") # YOLO 모델 설정
    model.train(data="data_pothole.yaml", epochs=12, batch=32, project='./pothole_detection_result/', device=[0, 1])
