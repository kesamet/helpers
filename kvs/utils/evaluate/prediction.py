"""
Predict and write to txt.
"""

import os

import cv2

from config import CFG
from utils.evaluate.image_extraction import uuid_names


def nxywh2xyxy(row):
    fw, fh = CFG.new_fsize
    cx, cy, w, h = row
    x0 = int((cx - w / 2) * fw)
    y0 = int((cy - h / 2) * fh)
    x1 = int((cx + w / 2) * fw)
    y1 = int((cy + h / 2) * fh)
    return x0, y0, x1, y1


def xyxy2nxywh(row):
    fw, fh = CFG.new_fsize
    x0, y0, x1, y1 = row
    cx = (x0 + x1) / 2 / fw
    cy = (y0 + y1) / 2 / fh
    w = (x1 - x0) / fw
    h = (y1 - y0) / fh
    return cx, cy, w, h


def predict_write(img_path, dest_path):
    """Detect objects and write the bounding boxes to a text file."""
    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imgs = process_img([frame])
    rects = obj_detection(imgs)

    if len(rects) == 0:
        return

    with open(dest_path, "w") as f:
        for rect in rects:
            cx, cy, w, h = xyxy2nxywh(rect[:4])
            f.write(f"{rect[4]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main():
    modelpath = "weights/best.pt"
    data_dir = "weights/dataset"
    camera_ids = [0, 1, 2, 3, 4]

    os.environ["modelpath"] = modelpath
    from utils.serve.detecting import process_img, obj_detection

    for camera_id in camera_ids:
        print("camera_id", camera_id)
        dest = f"{data_dir}/{uuid_names[camera_id]}/labels_v0/"
        if not os.path.isdir(dest):
            os.mkdir(dest)
        else:
            raise Exception(f"{dest} exists")

        for imgfn in os.listdir(f"{data_dir}/{uuid_names[camera_id]}/images/"):
            txtfn = imgfn.split(".")[0]
            predict_write(
                f"{data_dir}/{uuid_names[camera_id]}/images/{imgfn}", f"{dest}{txtfn}.txt"
            )


if __name__ == "__main__":
    main()
