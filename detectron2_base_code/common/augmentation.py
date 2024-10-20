import copy
import random
import cv2
import numpy as np
import torch
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

def apply_cutout(image, n_holes=1, length=50):
    """
    Apply Cutout to the given image.
    
    Args:
        image (numpy.ndarray): The input image on which to apply Cutout, assumed to be of shape (H, W, C).
        n_holes (int): Number of holes to cut out from the image.
        length (int): The length (in pixels) of each square hole.
        
    Returns:
        numpy.ndarray: The image with cutout applied.
    """
    h, w, c = image.shape
    
    # Create a mask with the same size as the image
    mask = np.ones((h, w), np.float32)
    
    for _ in range(n_holes):
        # Randomly choose the center of the square hole
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        # Calculate the top-left and bottom-right coordinates of the hole
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        
        # Zero out the pixels within the square
        mask[y1:y2, x1:x2] = 0.0
    
    # Expand the mask to all channels
    mask = np.expand_dims(mask, axis=-1)
    
    # Apply the mask to the image
    image = image * mask
    
    return image.astype(np.uint8)

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
    
    if dataset_dicts and random.random() < 0.25:
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
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomBrightness(0.8, 1.4),
            T.RandomContrast(0.6, 1.3),
            T.RandomCrop_CategoryAreaConstraint("absolute", (640, 640))
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

# Define the RandAugment pipeline using Albumentations
def get_randaug_transforms():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_id']))

# Custom Mapper with RandAugment
def MyAlbMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # Read the image
    image = utils.read_image(dataset_dict['file_name'], format="BGR")
    
    # Get annotations (bounding boxes and category labels)
    annos = dataset_dict['annotations']
    bboxes = [obj["bbox"] for obj in annos]
    category_ids = [obj["category_id"] for obj in annos]

    # Albumentations expects bounding boxes in [x_min, y_min, x_max, y_max] format (Pascal VOC)
    bboxes = [utils.convert_bbox_format(bbox, 'xywh', 'xyxy') for bbox in bboxes]

    # Apply RandAugment using Albumentations
    aug = get_randaug_transforms()
    augmented = aug(image=image, bboxes=bboxes, category_id=category_ids)
    image = augmented["image"].numpy()  # Get the augmented image
    bboxes = augmented["bboxes"]  # Get augmented bounding boxes
    category_ids = augmented["category_id"]  # Get augmented labels

    # Convert back to Detectron2 format
    annos = []
    for bbox, category_id in zip(bboxes, category_ids):
        annos.append({
            "bbox": utils.convert_bbox_format(bbox, 'xyxy', 'xywh'),
            "bbox_mode": dataset_dict['annotations'][0]['bbox_mode'],
            "category_id": category_id
        })

    # Update dataset_dict with augmented data
    dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))
    dataset_dict['annotations'] = annos
    
    return dataset_dict

def MyBaseMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomBrightness(0.8, 1.4),
            T.RandomContrast(0.6, 1.3),
            T.RandomCrop_CategoryAreaConstraint("absolute", (640, 640))
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)

    # Apply Cutout (for example, 1 hole of size 50)
    # image = apply_cutout(image, n_holes=1, length=50)

    dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict

def MyInfMapper(dataset_dict):
    """Mapper function for inference. No augmentations applied."""
    
    # Perform a deep copy
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # Read the image as is without augmentations
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    # Convert the image to a tensor (shape: [C, H, W])
    dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))

    # If annotations are provided, convert them to instances without any transformation
    if "annotations" in dataset_dict:
        annos = [
            utils.transform_instance_annotations(obj, None, image.shape[:2])
            for obj in dataset_dict.pop('annotations')
            if obj.get('iscrowd', 0) == 0
        ]

        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict
