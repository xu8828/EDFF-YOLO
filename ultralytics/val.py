import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/xie/xcl/paper/code/yolov8/ultralytics/runs/detect/ALL/weights/best.pt')
    model.val(data='/home/xie/xcl/paper/code/yolov8/ultralytics/ultralytics/cfg/datasets/ruod.yaml',
                split='test',
                save_json=True, # if you need to cal coco metrice
                project='runs/v8s',
                name='exp',
                batch=1,
                device='1',
                )