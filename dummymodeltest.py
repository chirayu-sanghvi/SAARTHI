import mmcv, torch
from PIL import Image
from mmdet.apis import DetInferencer
import glob
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from modelsalienttest import process_salient_image

base_costs = {
    0: 100,  # dents
    1: 50,   # scratches
    2: 200,  # cracks
    3: 150,  # lamp broken
    4: 80,   # tire flat
    5: 250   # glass shattered
}


def load_model():
    config_file = 'D:/Cardd/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_poly_cardamage_config.py'
    checkpoint_file = 'D:\Cardd\mmdetection\work_dirs\mask_rcnn_r50_caffe_fpn_poly_cardamage_config\mask_rcnn_r50_caffe_fpn_poly_cardamage_config\\best_coco_bbox_mAP_epoch_9.pth'

    inference_result = DetInferencer(config_file, checkpoint_file, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(inference_result)
    return inference_result

# model initialization
model = load_model()
def process_image(image_path, results_folder='D:/Cardd/static/results'):
    result = model(image_path, out_dir=results_folder)
    predictions = result['predictions'][0]
    labels = predictions['labels']
    scores = predictions['scores']
    bboxes = predictions['bboxes']
    print(scores)
    repair_cost = calculate_repair_cost(bboxes, labels, scores, base_costs, image_path, results_folder)
    print(f"Total estimated repair cost: ${repair_cost:.2f}")

    last_token = image_path.split('\\')[-1] 
    result_image_path = results_folder+ "\\"+last_token
    return f"{repair_cost}"


def compute_iou(boxA, boxB):
    # Calculate the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def non_maximum_suppression(boxes, scores, iou_threshold):
    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = np.argsort(scores)[::-1]
    print("I am box", boxes)
    print("I am scores",scores)
    print("I am index", idxs)
    selected_idxs = []

    while idxs.size > 0:
        current_idx = idxs[0]
        selected_idxs.append(current_idx)

        if idxs.size == 1:
            break

        # Calculate IoU with the rest
        rest_idxs = idxs[1:]
        rest_boxes = boxes[rest_idxs]
        ious = np.array([compute_iou(boxes[current_idx], rest_boxes[i]) for i in range(len(rest_idxs))])

        # Only keep boxes with IoU less than threshold
        idxs = rest_idxs[ious < iou_threshold]

    return selected_idxs

def calculate_repair_cost(boxes, labels, scores, base_costs, image_path, results_folder, normalization_factor=10000):
    # Apply Non-Maximum Suppression to filter boxes
    selected_indices = non_maximum_suppression(boxes, scores, 0.2)
    print("I am the image path", image_path)
    output_image = draw_bounding_boxes(image_path, boxes, scores, labels, selected_indices)
    last_token = image_path.split('\\')[-1] 
    cost_image_path = "\\"+last_token
    print("I am the print save folder ",results_folder+"cost\\"+cost_image_path)
    # Save or display the output image
    cv2.imwrite(results_folder+"\\"+"cost"+"\\"+cost_image_path, output_image)
    # cv2.imshow('Selected Detections', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    total_cost = 0
    count = 1
    # Calculate cost for each selected box
    for index in selected_indices:
        label = labels[index]
        score = scores[index]
        bbox = boxes[index]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Calculate area (width * height)
        
        base_cost = base_costs.get(label, 0)  # Get base cost for the damage type
        severity_adjustment = (area * score) / normalization_factor

        cost = base_cost * (1 + severity_adjustment)
        print(f"adding cost of {cost} for {label} - {count}")
        count = count + 1
        total_cost += cost

    return total_cost

def draw_bounding_boxes(image_path, boxes, scores, labels, selected_indices):
    # Load the image
    image = cv2.imread(image_path)
    
    # Define colors for different classes
    colors = {
        0: (0, 0, 255),  # Blue for class 0
        1: (0, 255, 0),  # Green for class 1
        2: (255, 0, 0),  # Red for class 2
        3: (0, 255, 255), # Cyan for class 3
        4: (0, 165, 255), # Orange for class 4
        5: (128, 0, 128)  # Purple for class 5
    }
    
    damage_label = {
        0: "Dent",
        1: "Scratch",
        2: "Crack",
        3: "Lamp Broken",
        4: "Tire Flat",
        5: "Glass Shatter"
    }
    
    # Draw each bounding box and add a counter and severity label
    for idx, index in enumerate(selected_indices):
        bbox = boxes[index]
        label = labels[index]
        score = scores[index]
        color = colors.get(label, (0, 255, 0))  # Default to green if label is unknown
        
        # Determine severity
        severity = "Major" if score >= 0.15 else "Minor"
        severity_text = f"{severity}"
        
        # Draw the bounding box filled with color
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color)  # -1 fills the rectangle
        
        # Add the severity text in the center of the box
        severity_text_size = cv2.getTextSize(severity_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        severity_text_x = int((bbox[0] + bbox[2]) / 2 - severity_text_size[0] / 2)
        severity_text_y = int((bbox[1] + bbox[3]) / 2 + severity_text_size[1] / 2)
        cv2.putText(image, severity_text, (severity_text_x, severity_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Prepare text for label and damage number
        image_label = damage_label.get(label, "Unknown")    
        label_text = f'{image_label}, Damage Number {idx+1}'
        
        # Calculate position for the label text
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = int(bbox[0])
        text_y = int(bbox[1] - 10 - text_size[1])
        
        # Draw black background for the text
        cv2.rectangle(image, (text_x, text_y), (text_x + text_size[0], text_y + text_size[1] + 10), (0,0,0), -1)
        
        # Put white text
        cv2.putText(image, label_text, (text_x, text_y + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image

