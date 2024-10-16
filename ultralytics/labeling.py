from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import os

image_width = 1024
image_height = 1024

train_coco = COCO('/data/ephemeral/home/PJU/dataset/train_split.json')
val_coco = COCO('/data/ephemeral/home/PJU/dataset/val_split.json')

train_output_dir = '/data/ephemeral/home/PJU/yolo_series/train/labels'
val_output_dir = '/data/ephemeral/home/PJU/yolo_series/val/labels'

train_df = pd.DataFrame()
val_df = pd.DataFrame()

classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
train_image_ids = [], val_image_ids = []
train_class_name = [], val_class_name = []
train_class_id = [], val_class_id = []
train_x_min = [], val_x_min = []
train_y_min = [], val_y_min = []
train_x_max = [], val_x_max = []
train_y_max = [], val_y_max = []

for image_id in train_coco.getImgIds():
        
    image_info = train_coco.loadImgs(image_id)[0]
    ann_ids = train_coco.getAnnIds(imgIds=image_info['id'])
    anns = train_coco.loadAnns(ann_ids)
        
    file_name = image_info['file_name']
    file_name = os.path.basename(file_name)
  
    for ann in anns:
        train_image_ids.append(file_name)
        train_class_name.append(classes[ann['category_id']])
        train_class_id.append(ann['category_id'])
        train_x_min.append(float(ann['bbox'][0]))
        train_y_min.append(float(ann['bbox'][1]))
        train_x_max.append(float(ann['bbox'][0]) + float(ann['bbox'][2]))
        train_y_max.append(float(ann['bbox'][1]) + float(ann['bbox'][3]))

train_df['image_id'] = train_image_ids
train_df['class_name'] = train_class_name
train_df['class_id'] = train_class_id
train_df['x_min'] = train_x_min
train_df['y_min'] = train_y_min
train_df['x_max'] = train_x_max
train_df['y_max'] = train_y_max

for image_id, group in train_df.groupby('image_id'):
    txt_file_name = os.path.join(train_output_dir, os.path.splitext(os.path.basename(image_id))[0] + '.txt')
    with open(txt_file_name, 'w') as f:
        for _, row in group.iterrows():
            class_id = row['class_id']
            
            # 바운딩 박스 중심 좌표와 너비 및 높이를 계산하고 정규화합니다.
            x_center = ((row['x_min'] + row['x_max']) / 2) / image_width
            y_center = ((row['y_min'] + row['y_max']) / 2) / image_height
            width = (row['x_max'] - row['x_min']) / image_width
            height = (row['y_max'] - row['y_min']) / image_height
            
            # YOLO 형식으로 기록합니다.
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

for image_id in val_coco.getImgIds():
        
    image_info = val_coco.loadImgs(image_id)[0]
    ann_ids = val_coco.getAnnIds(imgIds=image_info['id'])
    anns = val_coco.loadAnns(ann_ids)
        
    file_name = image_info['file_name']
    file_name = os.path.basename(file_name)
  
    for ann in anns:
        val_image_ids.append(file_name)
        val_class_name.append(classes[ann['category_id']])
        val_class_id.append(ann['category_id'])
        val_x_min.append(float(ann['bbox'][0]))
        val_y_min.append(float(ann['bbox'][1]))
        val_x_max.append(float(ann['bbox'][0]) + float(ann['bbox'][2]))
        val_y_max.append(float(ann['bbox'][1]) + float(ann['bbox'][3]))

val_df['image_id'] = val_image_ids
val_df['class_name'] = val_class_name
val_df['class_id'] = val_class_id
val_df['x_min'] = val_x_min
val_df['y_min'] = val_y_min
val_df['x_max'] = val_x_max
val_df['y_max'] = val_y_max

for image_id, group in val_df.groupby('image_id'):
    txt_file_name = os.path.join(val_output_dir, os.path.splitext(os.path.basename(image_id))[0] + '.txt')
    with open(txt_file_name, 'w') as f:
        for _, row in group.iterrows():
            class_id = row['class_id']
            
            # 바운딩 박스 중심 좌표와 너비 및 높이를 계산하고 정규화합니다.
            x_center = ((row['x_min'] + row['x_max']) / 2) / image_width
            y_center = ((row['y_min'] + row['y_max']) / 2) / image_height
            width = (row['x_max'] - row['x_min']) / image_width
            height = (row['y_max'] - row['y_min']) / image_height
            
            # YOLO 형식으로 기록합니다.
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")