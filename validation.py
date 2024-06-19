from ultralytics import YOLO
import os


if __name__ == '__main__':
    model = YOLO("/home/work/road_mark/ai_hub_learning/eqaulity_class_roadmark_seg_result/train_epoch_1000_p_5/weights/best.pt") # YOLO 모델 설정
    model.val(data="data_roadmark.yaml", batch=128, device=[0, 1])
