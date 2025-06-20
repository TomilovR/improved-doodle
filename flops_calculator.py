import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from thop import profile, clever_format

SLICE_HEIGHT = 1024
SLICE_WIDTH = 1024
OVERLAP_HEIGHT_RATIO = 0.25
OVERLAP_WIDTH_RATIO = 0.25
IMAGE_SIZE = 1024
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

logging.getLogger().setLevel(logging.WARNING)

def download_yolo_model(model_name: str = 'yolov8s.pt') -> str:
    model_path = Path(model_name)
    if not model_path.exists():
        model = YOLO(model_name)
    return str(model_path)

def load_model(model_path: str) -> YOLO:
    if not Path(model_path).exists():
        model_path = download_yolo_model(model_path)
    return YOLO(model_path)

def create_sahi_detection_model(model_path: str) -> AutoDetectionModel:
    if not Path(model_path).exists():
        if Path('best.pt').exists():
            model_path = 'best.pt'
        else:
            model_path = download_yolo_model('yolov8s.pt')
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=CONF_THRESHOLD,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return detection_model


def measure_yolo_flops(model: YOLO) -> float:
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    flops, _ = profile(model.model, inputs=(dummy_input,))
    flops, _ = clever_format([flops, _], "%.3f")
    
    if 'G' in flops:
        flops_num = float(flops.replace('G', '')) * 1e9
    elif 'M' in flops:
        flops_num = float(flops.replace('M', '')) * 1e6
    else:
        flops_num = float(flops)
    
    return flops_num


def measure_sahi_flops(model: YOLO, image_path: str) -> float:
    dummy_input = torch.randn(1, 3, SLICE_HEIGHT, SLICE_WIDTH)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    flops, _ = profile(model.model, inputs=(dummy_input,))
    flops, _ = clever_format([flops, _], "%.3f")
    
    if 'G' in flops:
        slice_flops = float(flops.replace('G', '')) * 1e9
    elif 'M' in flops:
        slice_flops = float(flops.replace('M', '')) * 1e6
    else:
        slice_flops = float(flops)
    
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    step_h = int(SLICE_HEIGHT * (1 - OVERLAP_HEIGHT_RATIO))
    step_w = int(SLICE_WIDTH * (1 - OVERLAP_WIDTH_RATIO))
    
    slices_h = int(np.ceil((h - SLICE_HEIGHT) / step_h) + 1) if h > SLICE_HEIGHT else 1
    slices_w = int(np.ceil((w - SLICE_WIDTH) / step_w) + 1) if w > SLICE_WIDTH else 1
    
    total_slices = slices_h * slices_w
    total_flops = slice_flops * total_slices
    
    return total_flops


def find_sample_images(sample_dir: str = "sample_images") -> List[str]:
    sample_path = Path(sample_dir)
    if not sample_path.exists():
        return []
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(sample_path.glob(ext))
    
    return [str(f) for f in image_files]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def load_annotations(annotations_file: str, image_name: str) -> List[Dict]:
    df = pd.read_csv(annotations_file)
    annotations = []
    
    for _, row in df.iterrows():
        if (pd.notna(row['filename']) and row['filename'] == image_name and 
            pd.notna(row['class']) and row['class'] == 'human'):
            annotations.append({
                'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                'class': 'human'
            })
    
    return annotations


def run_inference(model: YOLO, image_path: str) -> List[Dict]:
    results = model(image_path, conf=CONF_THRESHOLD, verbose=False)
    detections = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': 'human'
                })
    
    return detections


def calculate_map50(predictions: List[List[Dict]], ground_truths: List[List[Dict]]) -> float:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for preds, gts in zip(predictions, ground_truths):
        matched_gts = set()
        
        for pred in preds:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gts:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= 0.5:
                total_tp += 1
                matched_gts.add(best_gt_idx)
            else:
                total_fp += 1
        
        total_fn += len(gts) - len(matched_gts)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    return precision if precision > 0 else 0


def measure_map50(model: YOLO, sample_images: List[str]) -> float:
    annotations_file = "sample_images/_annotations.csv"
    if not Path(annotations_file).exists():
        return 0.0
    
    predictions = []
    ground_truths = []
    
    for image_path in sample_images[:5]:
        image_name = Path(image_path).name
        
        preds = run_inference(model, image_path)
        gts = load_annotations(annotations_file, image_name)
        
        if len(gts) > 0:  # Only use images with annotations
            predictions.append(preds)
            ground_truths.append(gts)
    
    if len(ground_truths) == 0:
        return 0.0
    
    return calculate_map50(predictions, ground_truths)


def main():
    print("YOLO FLOPS Calculator")
    
    model_path = 'best.pt' if Path('best.pt').exists() else 'yolov8s.pt'
    model = load_model(model_path)
    
    normal_flops = measure_yolo_flops(model)
    
    sample_images = find_sample_images()
    
    if not sample_images:
        dummy_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        dummy_path = "dummy_image.jpg"
        cv2.imwrite(dummy_path, dummy_image)
        sample_images = [dummy_path]
    test_image = sample_images[0]
    sahi_flops = measure_sahi_flops(model, test_image)
    
    map50 = measure_map50(model, sample_images)
    
    print(f"Normal YOLO FLOPS: {normal_flops:.2e}")
    print(f"SAHI FLOPS: {sahi_flops:.2e}")
    print(f"SAHI increase: {sahi_flops / normal_flops:.1f}x")
    if map50 > 0:
        print(f"mAP@0.5: {map50:.3f}")
    else:
        print("mAP@0.5: not available (no annotations)")
    
    if "dummy_image.jpg" in sample_images:
        os.remove("dummy_image.jpg")


if __name__ == "__main__":
    main()