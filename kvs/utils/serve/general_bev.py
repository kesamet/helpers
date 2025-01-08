"""
General utility functions used in bev tracker.
"""

import math

import cv2
import numpy as np
import pandas as pd


def make_divisible(x, divisor):
    """Returns x evenly divisble by divisor"""
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    """Verify img_size is a multiple of stride s"""
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print(
            "WARNING: --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size


def cal_resize_params(img, new_shape=(640, 640), s=32, auto=True, scaleFill=False, scaleup=True):
    """
    Calculate resizing parameters while keeping original height-to-width ratio,
    including new unpadded image size, scale ratio, paddings.
    """
    shape = img.shape[:2][::-1]  # width, height
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # resize image to grid size multiple rectangle
    x, y = check_img_size(new_shape[0], s=s), check_img_size(new_shape[1], s=s)
    new_shape = (x, y)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[0], new_shape[1])
        ratio = new_shape[0] / shape[0], new_shape[1] / shape[1]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    return new_unpad, ratio, (int(dw), int(dh))


def scale_one_coord(coord, scale, pad, resized=False):
    if not resized:
        coord *= scale
        coord += pad
    else:
        coord -= pad
        coord *= scale
    return coord


def non_max_suppression(df: pd.DataFrame, overlapThresh: float):
    ls = []
    for cid in df.cam.unique():
        sub = df[df.cam == cid].reset_index(drop=True)
        pick = []
        x0 = sub["x0"].values
        y0 = sub["y0"].values
        x1 = sub["x1"].values
        y1 = sub["y1"].values

        # compute the area of the bboxes and sort by the bottom-right y1
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        indexes = np.argsort(y1)
        while len(indexes) > 0:
            # pick from the last index
            last = len(indexes) - 1
            i = indexes[last]
            pick.append(i)

            # calculate the overlapping area with the picked bbox
            xx0 = np.maximum(x0[i], x0[indexes[:last]])
            yy0 = np.maximum(y0[i], y0[indexes[:last]])
            xx1 = np.minimum(x1[i], x1[indexes[:last]])
            yy1 = np.minimum(y1[i], y1[indexes[:last]])

            # compute overlapping ratio
            w = np.maximum(0, xx1 - xx0 + 1)
            h = np.maximum(0, yy1 - yy0 + 1)
            overlap = (w * h) / area[indexes[:last]]

            # delete the bboxes with overlapping ratio above threshold
            filtered = np.where(overlap > overlapThresh)[0].tolist()
            indexes = np.delete(indexes, [last] + filtered)
        ls.append(sub.loc[pick])
    # return detection dataframe with overlapping bboxes removed
    return pd.concat(ls, ignore_index=True)


def camera_to_world(camera: dict, imagePoints, z=0):
    objectPoints = cv2.undistortPoints(
        src=imagePoints,
        cameraMatrix=camera["cameraMatrix"],
        distCoeffs=camera["distCoeffs"],
    ).squeeze(axis=1)
    # Construct the normalized world coordinates, ie. (X/Z, Y/Z, 1)
    n_world = np.concatenate([objectPoints.T, np.ones(shape=(1, len(imagePoints)))])
    # Compute scaling factor assuming ground plane, ie. z = 0
    s = (z + camera["r_inv"][2] @ camera["tvec"]) / (camera["r_inv"][2] @ n_world)
    p_world = camera["r_inv"] @ (s * n_world - camera["tvec"])
    return p_world.T


def world_to_camera(camera: dict, objectPoints):
    projected = cv2.projectPoints(
        objectPoints=objectPoints,
        rvec=camera["rvec"],
        tvec=camera["tvec"],
        cameraMatrix=camera["cameraMatrix"],
        distCoeffs=camera["distCoeffs"],
    )
    return projected[0].squeeze(axis=1)


####################### For plotting ###########################
COLORS = [[0, 128, 255], [255, 0, 255]]
LABELS = ["standing", "lying down"]


def plot_bboxes(frames, rects, camera_ids):
    for i, cid in enumerate(camera_ids):
        if cid in rects.keys():
            for rect in rects[cid]:
                if len(rect) > 4:
                    plot_one_box(
                        frames[i],
                        rect[:4],
                        label=LABELS[int(rect[4])],
                        confidence=rect[5],
                        color=COLORS[int(rect[4])],
                        line_thickness=1,
                    )
                else:
                    plot_one_box(
                        frames[i],
                        rect,
                        line_thickness=1,
                    )


def plot_one_box(img, bbox, label=None, confidence=None, color=None, line_thickness=None):
    """Plots one bounding box on image."""
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or (255, 255, 255)
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(str(label), 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

        if confidence is None:
            cv2.putText(
                img=img,
                text=label,
                org=(c1[0], c1[1] - 2),
                fontFace=0,
                fontScale=tl / 3,
                color=[225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.putText(
                img=img,
                text=f"{label} {confidence:.2f}",
                org=(c1[0], c1[1] - 2),
                fontFace=0,
                fontScale=tl / 3,
                color=[225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )


def add_info(frames, object_id, obj_info, cameras):
    camera_ids = list(cameras.keys())
    for _, info in obj_info.iterrows():
        height = info.height
        camera = cameras[info.cam]
        frame = frames[camera_ids.index(info.cam)]
        # scale from original frame size to new frame size specified in config.py
        p_x = scale_one_coord(info.c_x, camera["ratio"], camera["pad"][0])
        p_y = scale_one_coord(info.y1, camera["ratio"], camera["pad"][1])
        p_x = np.clip(p_x, 0, camera["new_fsize"][0])
        p_y = np.clip(p_y, 0, camera["new_fsize"][1])

        cv2.putText(
            img=frame,
            text=f"{int(object_id)} {height:.2f}",
            org=(int(p_x) - 10, int(p_y) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1,
        )


# def add_info(frame, object_id, info, camera):
#     height = info[2]
#     proj_pos = info[3:5]
#     # scale from original frame size to new frame size specified in config.py
#     projx = scale_one_coord(proj_pos[0], camera['ratio'], camera['pad'][0])
#     projy = scale_one_coord(proj_pos[1], camera['ratio'], camera['pad'][1])
#     projx = np.clip(projx, 0, camera['new_fsize'][0])
#     projy = np.clip(projy, 0, camera['new_fsize'][1])

#     cv2.circle(
#         frame,
#         (int(projx), int(projy)),
#         4,
#         (0, 255, 0),
#         -1
#         )

#     cv2.putText(
#         frame,
#         f'{int(object_id)} {height:.2f}',
#         (int(projx) - 10, int(projy) - 10),
#         cv2.FONT_HERSHEY_SIMPLEX,dd
#         0.5,
#         (0, 255, 0),
#         1
#     )


def resize_frame(camera, img, interpolation=cv2.INTER_LINEAR):
    """Resize frame while keeping original height-to-width ratio"""
    if img.shape[:2] != camera["new_fsize"]:
        img = cv2.resize(img, camera["unpad_fsize"], interpolation=interpolation)
        dw, dh = camera["pad"]
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
    return img
