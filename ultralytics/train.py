import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from ultralytics import YOLO

# Load a model   V8S+URPC
model = YOLO('yolov8s-develop.yaml')  # 从YAML中构建一个新模型
# model = YOLO('yolov8n.pt')  #加载预训练的模型
model.info()
# Train the model
results = model.train(data='/home/xie/xcl/paper/code/yolov8/ultralytics/ultralytics/cfg/datasets/ruod.yaml', epochs=300, imgsz=640, batch=16, amp=False, cache=False,  close_mosaic=0,
             device='0',
            )
