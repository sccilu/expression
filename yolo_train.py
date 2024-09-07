import torch
from ultralytics import YOLO
import os

def train_model():
    # 设置CUDA环境
    device = torch.device("cuda")
    print(f"Using GPU: {device}")

    model = YOLO('yolov8n-p2.yaml').load('yolov8n.pt')
    # 进行模型训练
    model.train(
        # 从头开始训练
        data='data.yaml',
        epochs=50,
        imgsz=640,
        device=device,
        workers = 4,
        batch =64,
    )
    # 进行模型验证
    model.val()
if __name__ == "__main__":
    # 调用训练函数
    train_model()

    #  python yolo_train.py
