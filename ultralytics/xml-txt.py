# -*- coding: utf-8 -*-
# xml解析包
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2

sets = ['train', 'test', 'val']
# classes = ['greenplate','blueplate','face']
# classes = ['holothurian', 'echinus', 'scallop', 'starfish', 'waterweeds']  # urpc
classes = ['holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle',
           'jellyfish']  # ruod
images = '/home/xie/xcl/paper/code/yolov8/ultralytics/RUOD/images'


# 进行归一化操作
def convert(size, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    x = (box[0] + box[1]) / 2.0  # 物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 物体在图中的中心点y坐标
    w = box[1] - box[0]  # 物体实际像素宽度
    h = box[3] - box[2]  # 物体实际像素高度
    x = x * dw  # 物体中心点x的坐标比(相当于 x/原图w)
    w = w * dw  # 物体宽度的宽度比(相当于 w/原图w)
    y = y * dh  # 物体中心点y的坐标比(相当于 y/原图h)
    h = h * dh  # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]


# year ='2012', 对应图片的id（文件名）
def convert_annotation(image_id):
    # 对应的通过year 找到相应的文件夹，并且打开相应image_id的xml文件，其对应bund文件
    in_file = open('/home/xie/xcl/paper/code/yolov8/ultralytics/RUOD/Annotations/%s.xml' % (image_id), encoding='utf-8')
    # 准备在对应的image_id 中写入对应的label，分别为
    # <object-class> <x> <y> <width> <height>
    out_file = open('/home/xie/xcl/paper/code/yolov8/ultralytics/RUOD/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
    # 解析xml文件
    tree = ET.parse(in_file)
    # 获得对应的键值对s
    root = tree.getroot()
    # 获得图片的尺寸大小

    # size = root.find('size')
    # # 如果xml内的标记为空，增加判断条件
    # if size != None:
    #     # 获得宽
    #     w = int(size.find('width').text)
    #     # 获得高
    #     h = int(size.find('height').text)
    #     # 遍历目标obj
    #     for obj in root.iter('object'):
    #         # 获得difficult ？？
    #         difficult = obj.find('difficult').text
    #         # 获得类别 =string 类型
    #         cls = obj.find('name').text
    #         # 如果类别不是对应在我们预定好的class文件中，或difficult==1则跳过
    #         if cls not in classes or int(difficult) == 1:
    #             continue
    #         # 通过类别名称找到id
    #         cls_id = classes.index(cls)
    #         # 找到bndbox 对象
    #         xmlbox = obj.find('bndbox')
    #         # 获取对应的bndbox的数组 = ['xmin','xmax','ymin','ymax']
    #         b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
    #              float(xmlbox.find('ymax').text))
    #         print(image_id, cls, b)
    #         # 带入进行归一化操作
    #         # w = 宽, h = 高， b= bndbox的数组 = ['xmin','xmax','ymin','ymax']
    #         bb = convert((w, h), b)
    #         # bb 对应的是归一化后的(x,y,w,h)
    #         # 生成 calss x y w h 在label文件中
    #         out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    img = cv2.imread(images + "/" + image_id + '.jpg')
    # print(images)
    w = int(img.shape[1])
    h = int(img.shape[0])

    # # 遍历目标obj
    for obj in root.iter('object'):
        cls = obj.find('name').text
        # if cls = waterweeds
        if cls not in classes == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(
            str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 返回当前工作目录
wd = getcwd()
print(wd)

for image_set in sets:

    # 先找labels文件夹如果不存在则创建
    if not os.path.exists('/home/xie/xcl/paper/code/yolov8/ultralytics/RUOD/labels/'):
        os.makedirs('/home/xie/xcl/paper/code/yolov8/ultralytics/RUOD/labels/')
    # 读取在ImageSets/Main 中的train、test..等文件的内容
    # 包含对应的文件名称
    image_ids = open('/home/xie/xcl/paper/code/yolov8/ultralytics/RUOD/ImageSets/%s.txt' % (image_set)).read().strip().split()
    # 打开对应的2012_train.txt 文件对其进行写入准备
    list_file = open('/home/xie/xcl/paper/code/yolov8/ultralytics/RUOD/%s.txt' % (image_set), 'w')
    # 将对应的文件_id以及全路径写进去并换行
    for image_id in image_ids:
        list_file.write('/home/xie/xcl/paper/code/yolov8/ultralytics/RUOD/images/%s.jpg\n' % (image_id))
        # 调用  year = 年份  image_id = 对应的文件名_id
        convert_annotation(image_id)
    # 关闭文件
    list_file.close()