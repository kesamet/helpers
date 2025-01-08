"""
Extract images from video.
"""

import os
from datetime import timedelta

import cv2
import pandas as pd
import pytz

from config import CFG
from utils.serve.kinesis import camera_uuid, devices

uuid_names = [uuid.split("-")[0] for uuid in camera_uuid]


def extract_images(vinput, dest, timestamp, camera_id):
    """
    Extract images from video.

    :param vinput:
    :param dest: destination directory
    :param timestamp: start timestamp. To be used for naming images
    :param camera_id: camera ID. To be used for naming images
    """
    try:
        vcap = cv2.VideoCapture(vinput)
        success, frame = vcap.read()
        if success:
            # resize
            frame = cv2.resize(frame, CFG.new_fsize, interpolation=cv2.INTER_AREA)
            t = timestamp.strftime("%Y%m%d%H%M%S")
            cv2.imwrite(dest + f"c{camera_id}_{t}.png", frame)

        vcap.release()
    except Exception:
        print(f"  No image found for camera_id {camera_id} for {timestamp.isoformat()}")


def main():
    """
    After extracting images, check that each image contains elephants

    Annotation labels:
    - 0: standing
    - 1: lyingdown
    """
    data_dir = "weights/dataset"
    camera_ids = [0, 1, 2, 3, 4]
    start_time = "2021-04-15 00:00"  # SGT
    end_time = "2021-04-15 09:00"  # SGT
    freq = "2T"

    # Create folders
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    dests = list()
    for camera_id in camera_ids:
        cam_folder = f"{data_dir}/{uuid_names[camera_id]}/"
        if not os.path.isdir(cam_folder):
            os.mkdir(cam_folder)

        dest = cam_folder + "images/"
        if not os.path.isdir(dest):
            os.mkdir(dest)
        dests.append(dest)

    # Extract images
    ts_range = pd.date_range(start_time, end_time, freq=freq, closed="left", tz="Asia/Singapore")
    for t0 in ts_range:
        print(f"Downloading - {t0}...")
        for camera_id, dest in zip(camera_ids, dests):
            extract_images(
                vinput=devices[camera_id].get_stream_url(t0.astimezone(pytz.utc)),
                dest=dest,
                timestamp=t0,
                camera_id=camera_id,
            )


if __name__ == "__main__":
    main()
