"""
Tracking utility functions.
"""
from collections import OrderedDict

import dlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dist


def get_centroid(bbox):
    """
    Use the bounding box coordinates to derive the centroid.
    bbox in (startX, startY, endX, endY) format
    """
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


class CentroidTracker:
    """Centroid tracker object."""
    def __init__(self, max_disappeared=50, max_distance=50):
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

    def register(self, centroid, rect, ts):
        """Use next available object_id to register a new object."""
        print(f"[INFO] {ts}: centroid tracker registering object {self.next_object_id}")
        self.objects[self.next_object_id] = CentroidObj(centroid, rect)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id, ts):
        """Delete object_id from both dictionaries to deregister."""
        print(f"[INFO] {ts}: centroid tracker deregistering object {object_id}")
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects, ts):
        """Update centroid tracker."""
        # for no detections,
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # deregister object_id if max_disappeared is reached
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id, ts)

            return self.objects

        # input centroids for the current frame
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_rects = list()
        for i, rect in enumerate(rects):
            input_centroids[i] = get_centroid(rect[:4])
            input_rects.append(rect)

        # if not tracking any objects, register input centroids
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_rects[i], ts)

        # otherwise try to match input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = [v.centroid for v in self.objects.values()]

            # compute distance between each pair of object centroids and input centroids
            # and use Hungarian algorithm to match input centroid to  object centroid
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows, cols = linear_sum_assignment(D)
            # rows = D.min(axis=1).argsort()
            # cols = D.argmin(axis=1)[rows]

            # used_rows = set()
            # used_cols = set()
            for (row, col) in zip(rows, cols):
                # if row in used_rows or col in used_cols:
                #     continue

                # if distance between centroids > max_distance,
                # do not associate centroid to object
                if D[row, col] > self.max_distance:
                    continue

                # otherwise, grab the object_id for the current row,
                # set its new centroid, and reset the disappeared counter
                object_id = object_ids[row]
                self.objects[object_id] = CentroidObj(
                    input_centroids[col], input_rects[col])
                self.disappeared[object_id] = 0

                # used_rows.add(row)
                # used_cols.add(col)

            # compute both the row and column index we have NOT yet examined
            unused_rows = set(range(0, D.shape[0])).difference(rows)
            unused_cols = set(range(0, D.shape[1])).difference(cols)
            # unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            # unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unused_rows:
                    # grab the object_id for the corresponding row
                    # index and increment the disappeared counter
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    # deregister object if max_disappeared is reached
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id, ts)

            # otherwise, register remaining input centroids as trackable objects
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], input_rects[col], ts)

        return self.objects


class TrackableObject:
    def __init__(self, object_id, centroid_obj):
        self.object_id = object_id
        self.centroids = [centroid_obj.centroid]
        self.rects = [centroid_obj.rect]

    def add(self, centroid_obj):
        self.centroids.append(centroid_obj.centroid)
        self.rects.append(centroid_obj.rect)


class CentroidObj:
    def __init__(self, xy, rect):
        self.centroid = list(xy)
        self.rect = list(rect)


def update_tracks(tracks, objects):
    """Update tracks."""
    for object_id, centroid_obj in objects.items():
        to = tracks.get(object_id)
        if to is None:
            tracks[object_id] = TrackableObject(object_id, centroid_obj)
        else:
            to.add(centroid_obj)


def init_tracking(all_rects, rgbs):
    """Initialize trackers."""
    all_trackers = []
    for rects, rgb in zip(all_rects, rgbs):
        trackers = init_tracking_per_frame(rects, rgb)
        all_trackers.append(trackers)
    return all_trackers


def init_tracking_per_frame(rects, rgb):
    """Initialize trackers for each frame."""
    trackers = []
    for rect in rects:
        x0, y0, x1, y1, label, conf = rect

        # start the dlib correlation tracker
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(x0, y0, x1, y1)
        tracker.start_track(rgb, rect)
        trackers.append((tracker, label, conf))
    return trackers


def obj_tracking(all_trackers, rgbs):
    """Object tracking."""
    all_rects = []
    for trackers, rgb in zip(all_trackers, rgbs):
        rects = obj_tracking_per_frame(trackers, rgb)
        all_rects.append(rects)
    return all_rects


def obj_tracking_per_frame(trackers, rgb):
    """Object tracking for each frame."""
    rects = []
    for tracker, label, conf in trackers:
        # update the tracker and grab the updated position
        tracker.update(rgb)
        pos = tracker.get_position()

        x0 = int(pos.left())
        y0 = int(pos.top())
        x1 = int(pos.right())
        y1 = int(pos.bottom())
        rects.append((x0, y0, x1, y1, label, conf))
    return rects


# TODO: the following is for task_track_v2: pseudo global coords
def to_global_coords(dfs):
    # Cam 1: no change
    # df1 = dfs[0]

    # Cam 2
    df2 = dfs[1]
    df2["x0"] += 640
    df2["y0"] += 80
    df2["x1"] += 640
    df2["y1"] += 80

    # Cam 4
    df4 = dfs[2]
    df4["y0"] += 560
    df4["y1"] += 560
    return dfs


def to_local_coords(rect, cam_id):
    rect = [int(x) for x in rect]
    if cam_id == 1:
        return rect
    elif cam_id == 2:
        return [rect[0] - 640, rect[1] - 80, rect[2] - 640, rect[3] - 80]
    elif cam_id == 4:
        return [rect[0], rect[1] - 560, rect[2], rect[3] - 560]
    else:
        raise Exception("Unknown cam_id")


def overlap(bbox1, bbox2):
    iy = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    uy = max(bbox1[3], bbox2[3]) - min(bbox1[1], bbox2[1])
    return iy / uy


def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def union_rect(rect1, rect2):
    bbox1 = rect1[:4]
    bbox2 = rect2[:4]
    return [[
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3]),
        rect1[4],
        (rect1[5] + rect2[5]) / 2,
    ] + list(rect1)[6:]]


def get_rects(dfs, round_msec):
    rects1 = dfs[0].query(f"round_msec == {round_msec}").values
    rects2 = dfs[1].query(f"round_msec == {round_msec}").values
    rects4 = dfs[2].query(f"round_msec == {round_msec}").values

    bool1 = (rects1[:, 2] > 620)
    bool2 = (rects2[:, 0] < 660)

    if bool1.sum() == 0 or bool2.sum() == 0:
        return np.vstack([rects1, rects2, rects4])

    if bool1.sum() == 1 and bool2.sum() == 1:
        r1 = rects1[bool1][0]
        r2 = rects2[bool2][0]
        if overlap(r1[:4], r2[:4]) > 0 and r1[4] == r2[4]:
            rect = union_rect(r1, r2)
            return np.vstack([rects1[~bool1], rects2[~bool2], rect, rects4])

    D = list()
    for r1 in rects1[bool1]:
        d = list()
        for r2 in rects2[bool2]:
            # Penalty of 100 if labels are different
            d.append(overlap(r1[:4], r2[:4]) - 100 * (r1[4] != r2[4]))
        D.append(d)
    rows, cols = linear_sum_assignment(D)
    all_rects = np.vstack([rects1[~bool1], rects2[~bool2]])
    for row, col in zip(rows, cols):
        if D[row][col] > 0:
            rect = union_rect(rects1[bool1][row], rects2[bool2][col])
            all_rects = np.vstack([all_rects, rect, rects4])
    return all_rects
