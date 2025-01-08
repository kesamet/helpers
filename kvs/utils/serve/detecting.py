"""
Detection and tracking initialization
"""
from os import getenv
import numpy as np
import torch

from config import CFG
from utils.general import non_max_suppression

MODELPATH = getenv("modelpath") or CFG.weights
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
print("Model path:", MODELPATH)
if DEVICE.type == "cpu":
    MODEL = torch.load(MODELPATH, map_location=DEVICE)['model'].float().fuse().eval()
    print("torch.backends.cudnn.benchmark =", torch.backends.cudnn.benchmark)
else:
    MODEL = torch.load(MODELPATH, map_location=DEVICE)['model'].float().fuse().half().eval()
    torch.backends.cudnn.benchmark = True

IMGSZ = CFG.imgsz
NAMES = MODEL.module.names if hasattr(MODEL, 'module') else MODEL.names
COLORS = [(0, 255, 255), (255, 0, 255)]


def process_img(img0):
    """Process image for YOLO."""
    img0 = np.asarray(img0)
    if len(img0.shape) == 3:
        img0 = np.expand_dims(img0, 0)

    # img = img0[:, :, :, ::-1]  # BGR to RGB, already done
    img = img0.transpose((0, 3, 1, 2))  # to NCHW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(DEVICE)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img


@torch.no_grad()
def obj_detection(imgs):
    """Object detection."""
    if DEVICE.type == "cpu":
        preds = MODEL(imgs, augment=CFG.augment)[0]
    else:
        preds = MODEL(imgs.half(), augment=CFG.augment)[0]

    # Apply NMS
    all_raw_dets = non_max_suppression(
        preds,
        CFG.conf_thres,
        CFG.iou_thres,
        classes=None,
        agnostic=CFG.agnostic_nms,
    )

    all_rects = []
    for raw_dets in all_raw_dets:
        raw_dets = raw_dets.detach().cpu().numpy()
        rects = []
        for startX, startY, endX, endY, conf, label in raw_dets:
            # filter out weak detections
            if conf > CFG.conf_thres:
                rects.append((int(startX), int(startY), int(endX), int(endY), int(label), conf))
        all_rects.append(rects)
    if len(all_rects) == 1:
        return all_rects[0]
    return all_rects
