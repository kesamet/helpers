import numpy as np


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    Returns:
        iou: float
    """
    # Get the coordinates of intersecting rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou


def evaluate_detections(pred_boxes, true_boxes, iou_threshold=0.5):
    """Evaluate object detection predictions.
    
    Args:
        pred_boxes: List of predicted boxes [[x1,y1,x2,y2,conf,class_id], ...]
        true_boxes: List of ground truth boxes [[x1,y1,x2,y2,class_id], ...]
        iou_threshold: IoU threshold for considering a positive detection
    
    Returns:
        dict: Dictionary containing precision, recall, and mAP
    """
    true_positives = 0
    false_positives = 0
    false_negatives = len(true_boxes)
    
    # Sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    
    # Match predictions to ground truth
    matched_gt = set()
    
    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth box
        for i, gt in enumerate(true_boxes):
            if i in matched_gt:
                continue
                
            if pred[5] != gt[4]:  # Check if same class
                continue
                
            iou = calculate_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        # Check if match found
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            false_negatives -= 1
        else:
            false_positives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def calculate_ap(recalls, precisions):
    """Calculate Average Precision using the 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_detections(pred_boxes, true_boxes, iou_thresholds=[0.5]):
    """Evaluate object detection predictions with multiple IoU thresholds.
    
    Args:
        pred_boxes: List of predicted boxes [[x1,y1,x2,y2,conf,class_id], ...]
        true_boxes: List of ground truth boxes [[x1,y1,x2,y2,class_id], ...]
        iou_thresholds: List of IoU thresholds for evaluation
    
    Returns:
        dict: Dictionary containing precision, recall, and mAP metrics
    """
    # Get unique classes
    pred_classes = set(int(box[5]) for box in pred_boxes)
    true_classes = set(int(box[4]) for box in true_boxes)
    unique_classes = sorted(pred_classes.union(true_classes))
    
    # Initialize metrics
    aps_by_threshold = {iou_th: [] for iou_th in iou_thresholds}
    
    # Sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    
    # Calculate AP for each class and IoU threshold
    for class_id in unique_classes:
        # Filter boxes by class
        class_pred_boxes = [box for box in pred_boxes if box[5] == class_id]
        class_true_boxes = [box for box in true_boxes if box[4] == class_id]
        
        for iou_threshold in iou_thresholds:
            # Initialize precision-recall curve
            num_true = len(class_true_boxes)
            num_pred = len(class_pred_boxes)
            
            if num_true == 0 and num_pred == 0:
                continue
            
            # Initialize arrays for precision-recall curve
            tp = np.zeros(num_pred)
            fp = np.zeros(num_pred)
            matched_gt = set()
            
            # Match predictions to ground truth
            for pred_idx, pred in enumerate(class_pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(class_true_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = calculate_iou(pred[:4], gt[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    tp[pred_idx] = 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp[pred_idx] = 1
            
            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / num_true if num_true > 0 else np.zeros_like(tp_cumsum)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Add sentinel values for AP calculation
            recalls = np.concatenate(([0.], recalls, [1.]))
            precisions = np.concatenate(([0.], precisions, [0.]))
            
            # Compute AP for this class and IoU threshold
            ap = calculate_ap(recalls, precisions)
            aps_by_threshold[iou_threshold].append(ap)
    
    # Calculate final metrics
    results = {}
    
    # Calculate mAP for each IoU threshold
    for iou_th in iou_thresholds:
        if len(aps_by_threshold[iou_th]) > 0:
            results[f'mAP@{iou_th}'] = np.mean(aps_by_threshold[iou_th])
        else:
            results[f'mAP@{iou_th}'] = 0.0
    
    # Calculate COCO-style mAP@0.5:0.95
    if len(iou_thresholds) > 1:
        all_maps = [results[f'mAP@{iou_th}'] for iou_th in iou_thresholds]
        results['mAP@0.5:0.95'] = np.mean(all_maps)
    
    return results


# def evaluate_model(model, dataloader, device):
#     """Evaluate model on a dataset.
    
#     Args:
#         model: Detection model
#         dataloader: DataLoader containing validation/test data
#         device: torch device
#     """
#     all_pred_boxes = []
#     all_true_boxes = []
    
#     model.eval()
#     with torch.no_grad():
#         for images, targets in dataloader:
#             images = images.to(device)
#             predictions = model(images)
            
#             # Convert predictions and targets to the required format
#             # This part depends on your model's output format
#             pred_boxes = convert_predictions(predictions)
#             true_boxes = convert_targets(targets)
            
#             all_pred_boxes.extend(pred_boxes)
#             all_true_boxes.extend(true_boxes)
    
#     # Define IoU thresholds for evaluation
#     iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.5, 0.55, ..., 0.95]
    
#     # Evaluate detections
#     results = evaluate_detections(all_pred_boxes, all_true_boxes, iou_thresholds)
    
#     return results
