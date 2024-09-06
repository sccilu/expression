import torch
from ultralytics import YOLO

def train_model():

    model = YOLO('yolov8n-p2.yaml').load('yolov8n.pt')
    #model = YOLO('runs/detect/train/weights/last.pt')
    # 进行模型训练
    model.train(
	#从头开始训练
        data='data.yaml',
        epochs=10,
        imgsz=640,
        device='cpu',
        workers = 8,
        batch =4,
    )

    # 进行模型验证s
    model.val()
if __name__ == "__main__":
    # 调用训练函数
    train_model()

    #  python yolo_train.py
