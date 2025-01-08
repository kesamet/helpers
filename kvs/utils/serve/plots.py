"""
Utility functions for plotting.
"""
from datetime import datetime

import cv2
import pytz

from config import CFG
from utils.serve.kinesis import devices


LABELS = [
    "Standing",
    "Lying down",
]
COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
]


def get_frame(cam, timestamp):
    """Get frame from device cam."""
    vcap = cv2.VideoCapture(devices[int(cam)].get_stream_url(timestamp.astimezone(pytz.utc)))
    success, frame = vcap.read()
    if success:
        resized_frame = cv2.resize(frame, CFG.new_fsize, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        vcap.release()
        return rgb


def plot_bboxes(frame, detections, font_scale=0.25, thickness=1):
    for x0, y0, x1, y1, label, conf in detections:
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)
        label = int(label)

        cv2.rectangle(
            img=frame,
            pt1=(x0, y0),
            pt2=(x1, y1),
            color=COLORS[label],
            thickness=1,
        )

        text = f"{LABELS[label]}: {conf:.3f}"
        t_size = cv2.getTextSize(text, 0, fontScale=font_scale, thickness=thickness)[0]
        c1 = (x0, y0)
        c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(frame, c1, c2, COLORS[label], -1, cv2.LINE_AA)
        cv2.putText(
            img=frame,
            text=text,
            org=(x0, y0 - 2),
            fontFace=0,
            fontScale=font_scale,
            color=(255, 255, 255),  # COLORS[label],
            thickness=thickness,
            lineType=2,
        )


def plot_points(frame, detections, font_scale=0.25, thickness=1):
    for object_id, x0, y0, x1, y1, label, conf in detections:
        xc = int((x0 + x1) / 2)
        yc = int((y0 + y1) / 2)
        label = int(label)

        cv2.circle(
            img=frame,
            center=(xc, yc),
            radius=4,
            color=COLORS[label],
            thickness=-1,
        )

        cv2.putText(
            img=frame,
            text=f"{object_id} {LABELS[label]}: {conf:.3f}",
            org=(xc, yc - 12),
            fontFace=0,
            fontScale=font_scale,
            color=COLORS[label],
            thickness=thickness,
            lineType=2,
        )


def add_info(frame, info, font_scale=0.6, thickness=2):
    for i, text in enumerate(info):
        cv2.putText(
            img=frame,
            text=text,
            org=(10, (i + 1) * 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
        )


def define_writer(vid_output, vid_fps, vid_fsize):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(vid_output, fourcc, vid_fps, vid_fsize, True)
    return writer
