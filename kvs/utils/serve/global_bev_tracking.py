"""
Tracking
"""
import copy
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment

from utils.serve.general_bev import cal_resize_params, camera_to_world


def resize_frame_helper(cameras: Dict[int, Dict], new_fsize: Tuple):
    """
    Resize frame helper function that generates ratio, padding, unpad frame size
    and frame size after padding provided original and new frame size, while
    keeping original frame height-to-width ratio.
    """
    for _, camera in cameras.items():
        w, h = camera["fsize"]
        frame = np.ones((h, w, 3))
        unpad_fsize, ratio, (dw, dh) = cal_resize_params(frame, new_shape=new_fsize)
        new_w, new_h = unpad_fsize
        camera["fsize"] = (w, h)
        camera["new_fsize"] = (new_w + 2 * dw, new_h + 2 * dh)
        camera["unpad_fsize"] = unpad_fsize
        camera["ratio"] = ratio[np.argmax([w, h])]
        camera["pad"] = (dw, dh)
    return cameras


def scale_bbox(all_rects: Dict, cameras: Dict):
    rects_copy = copy.deepcopy(all_rects)
    for cid, rects in all_rects.items():
        if len(rects) == 0:
            continue
        df = pd.DataFrame(rects)
        df.loc[:, 0:4:2] -= cameras[cid]["pad"][0]
        df.loc[:, 1:4:2] -= cameras[cid]["pad"][1]
        df.loc[:, :4] /= cameras[cid]["ratio"]
        rects_copy[cid] = df.values.tolist()
    return rects_copy


def scale_dets(dets_df: pd.DataFrame, cameras: Dict):
    """Reverse image coordinates scaling by pad and ratio."""
    df = dets_df.copy()
    for cid in df.cam.unique():
        camera = cameras[cid]
        df.loc[df.cam == cid, ["x0", "x1"]] -= camera["pad"][0]
        df.loc[df.cam == cid, ["y0", "y1"]] -= camera["pad"][1]
        df.loc[df.cam == cid, ["x0", "y0", "x1", "y1"]] /= camera["ratio"]
    return df


def _get_line(points):
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    return m, c


def set_sand_yard_boundary(df: pd.DataFrame):
    # sand yard boundary defined by lines connecting edge points
    m1, c1 = _get_line([[250, 1080], [1180, 900]])
    m2, c2 = _get_line([[1180, 900], [2180, 1050]])
    ylim = np.where(df["c_x"] <= 1180, m1 * df["c_x"] + c1, m2 * df["c_x"] + c2)
    df["y1"].clip(lower=ylim, inplace=True)
    return df


def process_detections(df: pd.DataFrame, cameras: Dict, xlim=(-5.5, 19), ylim=(-3, 11)):
    # get bottom edge to locate the ground plane position
    df["c_x"] = (df["x0"] + df["x1"]) / 2
    # get top edge to locate the projected head position in image space
    df["top_edge"] = list(zip(df["c_x"], df["y0"]))
    df_ls = list()
    for cid in df["cam"].unique():
        subset = df.query("cam == @cid").reset_index(drop=True)
        if cid == 4:
            # set boundary for sand yard
            subset = set_sand_yard_boundary(subset)

        # Computes the world space coordinate for detected targets
        camera = cameras[cid]
        p_image = subset[["c_x", "y1"]].to_numpy()
        p_world = pd.DataFrame(
            (camera_to_world(camera, p_image).squeeze() + camera["offset"]).reshape(-1, 3),
            columns=["wx", "wy", "wz"],
        )

        subset = pd.concat([subset, p_world], axis=1)
        if cid < 4:
            # trim wx, wy according to ground plane boundary
            subset["wx"] = subset["wx"].clip(lower=xlim[0], upper=xlim[1])
            subset["wy"] = subset["wy"].clip(lower=ylim[0], upper=ylim[1])

        df_ls.append(subset)
    return pd.concat(df_ls, ignore_index=True)


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=2):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.max_disappeared = max_disappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.max_distance = max_distance

    def register(self, centroid: list, ts: str):
        """Use next available object_id to register a new object."""
        print(f"[INFO] {ts}: tracker registering object {self.next_object_id}")
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int, ts: str):
        """Delete object_id from both dictionaries to deregister."""
        print(f"[INFO] {ts}: tracker deregistering object {object_id}")
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, positions: np.array, ts: str):
        # for no detections,
        if len(positions) == 0:
            # loop over any existing tracked objects and mark them as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # deregister object if reached max_disappeared
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id, ts)

        # initialize input centroids for the current frame
        input_centroids = np.copy(positions)
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], ts)
        # otherwise, try to match the input centroids to existing object centroids
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            input_centroids = list(input_centroids)

            D = dist.cdist(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            while True:
                row = rows[0]
                col = cols[0]

                # if distance between centroids > max_distance,
                # don't associate the two centroids to the same object
                if D[row, col] > self.max_distance:
                    break

                # otherwise, grab the object_id for the current row,
                # set its new centroid and reset disappeared counter
                self.objects[object_ids[row]] = input_centroids[col]
                self.disappeared[object_ids[row]] = 0

                # remove used object and its centroid
                object_ids.pop(row)
                object_centroids.pop(row)
                input_centroids.pop(col)

                if len(object_centroids) == 0 or len(input_centroids) == 0:
                    break

                new_D = dist.cdist(object_centroids, input_centroids)
                if D.shape == new_D.shape:
                    break

                D = new_D
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

            # for remaining object_ids, increase the disappeared counter
            for object_id in object_ids:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id, ts)

            # register remaining input centroids as trackable objects
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], ts)


class CentroidObj:
    def __init__(self, centroid: np.array, info: pd.DataFrame):
        self.centroid = centroid
        self.info = info


class TrackableObject:
    def __init__(self, object_id: int, centroid_obj: CentroidObj):
        # store object ID, centroids, other values stored in info
        self.objectid = object_id
        self.centroids = [centroid_obj.centroid]
        self.info = [centroid_obj.info]

    def add(self, centroid_obj: CentroidObj):
        self.centroids.append(centroid_obj.centroid)
        self.info.append(centroid_obj.info)


def get_bev_position_info(df: pd.DataFrame, bbox_max_distance=5):
    """
    Merges detections from multiple cameras according to projected position
    on global ground plane.
    p_fuse stores object positions on the ground plane.
    p_info's key is average position of associated detections on global
    ground plane, value is dataframe indices of associated detections.
    """
    camera_ids = sorted(df.cam.unique())
    if len(camera_ids) == 1:
        p_fuse = np.round(df[["wx", "wy"]].values, 3)
        p_info = {tuple(p_fuse[i]): [i] for i in range(len(p_fuse))}
    else:
        groups = df.groupby("cam").groups
        data_indices = groups[camera_ids[0]]
        info = {i: [data_indices[i]] for i in range(len(data_indices))}
        obj_id = len(data_indices) - 1
        p_update = df[df.cam == camera_ids[0]][["wx", "wy"]].values

        for cid in camera_ids[1:]:
            p_tmp = df[df.cam == cid][["wx", "wy"]].values
            D = dist.cdist(p_update, p_tmp)
            # if min distance to new bboxes of any existing bbox > bbox_max_distance,
            # avoid using this bbox for associating new bboxes
            indexes = np.where(D.min(axis=1) <= bbox_max_distance)[0]
            if 0 < len(indexes) < len(p_update):
                D = dist.cdist(p_update[indexes], p_tmp)

            rows, cols = linear_sum_assignment(D)
            for (row, col) in zip(rows, cols):
                # do not associate if distance between detections > bbox_max_distance
                if D[row, col] > bbox_max_distance:
                    p_update = np.vstack([p_update, p_tmp[col]])
                    # register new object
                    obj_id += 1
                    info[obj_id] = [groups[cid][col]]
                    continue
                # otherwise, update existing object position
                p_update[indexes[row]] = p_tmp[col]
                info[indexes[row]].append(groups[cid][col])

            # compute column index we have NOT yet examined
            unused_cols = set(range(0, D.shape[1])).difference(cols)
            for col in unused_cols:
                p_update = np.vstack([p_update, p_tmp[col]])
                obj_id += 1
                info[obj_id] = [groups[cid][col]]

        p_info = {}
        for indices in info.values():
            pos = np.round(df.loc[indices, ["wx", "wy"]].mean(), 3)
            p_info[tuple(pos)] = indices
        p_fuse = np.array(list(p_info.keys()))
    return p_fuse, p_info


def link_info_with_object(
        df: pd.DataFrame,
        pos_info: Dict[Tuple, List],
        centroid_tracker: CentroidTracker,
        timestamp: datetime,
    ):
    objs = {}
    for object_id, centroid in centroid_tracker.objects.items():
        if centroid_tracker.disappeared[object_id] == 0:
            obj = CentroidObj(
                [timestamp] + list(centroid),
                df.loc[pos_info[tuple(centroid)]],
            )
            objs[object_id] = obj
    return objs


def update_trackable_objects(
        trackable_objects: Dict[int, TrackableObject],
        centroid_tracker: CentroidTracker,
        centroid_objs: Dict[int, CentroidObj],
    ):
    """
    Update existing or register new trackable objects, update info
    including centroids, detections, world positions and height estimates.
    """
    for object_id in centroid_tracker.objects.keys():
        if centroid_objs.get(object_id) is None:
            continue
        to = trackable_objects.get(object_id)
        if to is None:
            trackable_objects[object_id] = TrackableObject(
                object_id, centroid_objs[object_id])
        else:
            to.add(centroid_objs[object_id])


def generate_tracking_results(trackable_objects: Dict[int, TrackableObject]):
    """
    Concatenate all trackable objects' centroid, camera, height, bbox, label,
    confidence etc in one dataframe
    """
    df_ls = []
    for obj in trackable_objects.keys():
        df = pd.DataFrame(trackable_objects[obj].centroids, columns=["timestamp", "cx", "cy"])
        df["object_id"] = obj

        info_df = pd.concat(
            [pd.DataFrame(x) for x in trackable_objects[obj].info],
            ignore_index=True
        )
        # info_df.columns = ["timestamp", "cam", "height", "projx", "projy",
        #                    "x0", "y0", "x1", "y1", "label", "confidence"]
        df = df.merge(info_df, on="timestamp", how="left")
        df_ls.append(df)

    return pd.concat(df_ls, ignore_index=True)


def scale_coords(
        df: pd.DataFrame,
        cameras: Dict[int, Dict],
        norm_coords=False,
        rescale_coords=True,
    ):
    """Normalize / rescale coordinates."""
    for cid in df.cam.unique():
        camera = cameras[cid]
        w, h = camera["fsize"]
        dw, dh = camera["pad"]
        ratio = camera["ratio"]

        xcols = ["x0", "x1"]
        ycols = ["y0", "y1"]
        df[xcols] = df[xcols].clip(lower=0, upper=w)
        df[ycols] = df[ycols].clip(lower=0, upper=h)

        if norm_coords:
            df.loc[df.cam == cid, xcols] /= w
            df.loc[df.cam == cid, ycols] /= h

        if rescale_coords:
            df.loc[df.cam == cid, xcols] *= ratio
            df.loc[df.cam == cid, xcols] += dw
            df.loc[df.cam == cid, ycols] *= ratio
            df.loc[df.cam == cid, ycols] += dh
    return df
