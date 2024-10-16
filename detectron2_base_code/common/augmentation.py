import copy
import random
import cv2
import numpy as np
import torch
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

def mosaic_augmentation(dataset_dicts, img_size=640):
    """Applies Mosaic Augmentation on four random images."""
    
    # Select 4 random images for mosaic
    indices = random.sample(range(len(dataset_dicts)), 4)
    images = []
    bboxes = []
    labels = []
    
    for idx in indices:
        # Read image
        img_dict = dataset_dicts[idx]
        img = utils.read_image(img_dict["file_name"], format="BGR")
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Resize the image
        h, w, _ = img.shape
        img = cv2.resize(img, (img_size // 2, img_size // 2))

        images.append(img)
        
        # Transform the bounding boxes
        for anno in img_dict["annotations"]:
            bbox = anno["bbox"]  # Format: [x_min, y_min, width, height]
            # Scale bounding boxes
            bbox = [bbox[0] * (img_size // w), bbox[1] * (img_size // h), bbox[2] * (img_size // w), bbox[3] * (img_size // h)]
            bboxes.append(bbox)
            labels.append(anno["category_id"])
    
    # Create an empty canvas for the mosaic
    mosaic_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Place the four images on the canvas
    mosaic_image[:img_size // 2, :img_size // 2, :] = images[0]  # Top-left
    mosaic_image[:img_size // 2, img_size // 2:, :] = images[1]  # Top-right
    mosaic_image[img_size // 2:, :img_size // 2, :] = images[2]  # Bottom-left
    mosaic_image[img_size // 2:, img_size // 2:, :] = images[3]  # Bottom-right
    
    return mosaic_image, bboxes, labels

def MyMapper(dataset_dict, dataset_dicts=None):
    """Mapper function with Mosaic augmentation and additional transforms."""
    
    # Perform a deep copy
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # If dataset_dicts is provided, apply mosaic augmentation 50% of the time
    if dataset_dicts and random.random() < 0.3:
        image, bboxes, labels = mosaic_augmentation(dataset_dicts)
        
        # Convert to tensor
        dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))
        
        # Generate annotations for the mosaic
        annotations = []
        for i, bbox in enumerate(bboxes):
            annotation = {
                "bbox": bbox,
                "bbox_mode": dataset_dict["annotations"][0]["bbox_mode"],  # Adjust as per your annotation format
                "category_id": labels[i]
            }
            annotations.append(annotation)
        
        dataset_dict["annotations"] = annotations
    else:
        # Normal augmentation pipeline
        image = utils.read_image(dataset_dict['file_name'], format='BGR')
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        transform_list = [
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomBrightness(0.8, 1.4),
            T.RandomContrast(0.6, 1.3)
        ]
        
        image, transforms = T.apply_transform_gens(transform_list, image)

        dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop('annotations')
            if obj.get('iscrowd', 0) == 0
        ]

        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict

def MyBaseMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict
