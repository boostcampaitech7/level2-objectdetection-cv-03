import copy
import random
import cv2
import numpy as np
import torch
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

def apply_CLAHE(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image."""
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final_img

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
    """Applies Mosaic Augmentation on four random images with bounding box adjustment and scaling."""
    
    # Select 4 random images for mosaic
    indices = random.sample(range(len(dataset_dicts)), 4)
    images = []
    bboxes = []
    labels = []
    
    for idx in indices:
        # Read image
        img_dict = dataset_dicts[idx]
        img = utils.read_image(img_dict["file_name"], format="BGR")
        
        # Random scaling factor to resize the image
        scale_factor = random.uniform(0.5, 1.5)
        
        # Resize the image
        h, w, _ = img.shape
        new_w = int(img_size // 2 * scale_factor)
        new_h = int(img_size // 2 * scale_factor)
        img = cv2.resize(img, (new_w, new_h))

        # Apply random color augmentation (brightness, contrast, etc.)
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.7, 1.3), beta=random.randint(-20, 20))

        images.append(img)
        
        # Adjust the bounding boxes and scale
        for anno in img_dict["annotations"]:
            bbox = anno["bbox"]  # Format: [x_min, y_min, width, height]
            # Scale the bounding boxes according to the new image size
            bbox = [
                bbox[0] * (new_w / w),
                bbox[1] * (new_h / h),
                bbox[2] * (new_w / w),
                bbox[3] * (new_h / h)
            ]
            bboxes.append(bbox)
            labels.append(anno["category_id"])
    
    # Create an empty canvas for the mosaic
    mosaic_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Place the four images on the canvas and adjust bounding boxes accordingly
    # Top-left (no offset needed)
    mosaic_image[:new_h, :new_w, :] = images[0]
    
    # Top-right (offset x by half of the img_size)
    mosaic_image[:new_h, img_size//2:img_size//2 + new_w, :] = images[1]
    for bbox in bboxes[len(images[0]["annotations"]):]:
        bbox[0] += img_size // 2  # Shift x-coordinates to the right
    
    # Bottom-left (offset y by half of the img_size)
    mosaic_image[img_size//2:img_size//2 + new_h, :new_w, :] = images[2]
    for bbox in bboxes[len(images[0]["annotations"]) + len(images[1]["annotations"]):]:
        bbox[1] += img_size // 2  # Shift y-coordinates down
    
    # Bottom-right (offset both x and y by half of the img_size)
    mosaic_image[img_size//2:img_size//2 + new_h, img_size//2:img_size//2 + new_w, :] = images[3]
    for bbox in bboxes[len(images[0]["annotations"]) + len(images[1]["annotations"]) + len(images[2]["annotations"]):]:
        bbox[0] += img_size // 2  # Shift x-coordinates to the right
        bbox[1] += img_size // 2  # Shift y-coordinates down
    
    # Clip bounding boxes to ensure they stay within the mosaic boundaries
    for bbox in bboxes:
        bbox[0] = max(0, min(bbox[0], img_size))
        bbox[1] = max(0, min(bbox[1], img_size))
        bbox[2] = max(0, min(bbox[2], img_size))
        bbox[3] = max(0, min(bbox[3], img_size))
    
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

def MyBaseMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    # Apply CLAHE augmentation
    image = apply_CLAHE(image, clip_limit=2.0, tile_grid_size=(8, 8))
    
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

    # Convert to tensor
    dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))

    # Transform and filter annotations
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
