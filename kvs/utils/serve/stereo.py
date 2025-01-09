import json
from datetime import datetime
from itertools import combinations
from typing import Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.store.s3 import AWSCredentials, S3PredictionStore

columns = [
    "timestamp",
    "x0",
    "y0",
    "x1",
    "y1",
    "label",
    "conf",
    "cam",
]


class Stereo:
    def __init__(self, num_targets: int):
        # Discretize world space by (cols, rows)
        self.grid_size = (48, 20)
        self.num_targets = num_targets

        # Load calibration parameters
        with open("./data/camera_sub.json", "r") as fp:
            camera = json.load(fp)
        self.cameraMatrix = np.array(camera["cameraMatrix"])
        self.distCoeffs = np.array(camera["distCoeffs"])

        # Set ground plane density to 2 points per meter
        self.density = 2
        # world space coordinate => meters in real world
        self.ground_plane = (
            np.array(
                [
                    # Follows C-style unravel_index
                    [x - 11, y - 1, 0]
                    for x in range(self.grid_size[0])
                    for y in range(self.grid_size[1])
                ],
                dtype=np.float32,
            )
            / self.density
        )

        # Transform ground plane to camera space
        # world space coordinate => pixels in camera space
        self.projections = [
            # Manually exclude points that are outside the camera space
            self.project_world(0, self.ground_plane[self.ground_plane[:, 0] >= -3]),
            self.project_world(1, self.ground_plane[self.ground_plane[:, 0] <= 9]),
        ]
        # world space coordinate => likelihood of detections
        self.occupancy_map = np.ones(shape=(len(self.projections),) + self.grid_size)
        self.occupancy_map /= self.grid_size[0] * self.grid_size[1]
        self.target_height = np.zeros(shape=(len(self.projections), self.num_targets))
        self.threshold = 0.5  # for filtering target positions on occupancy maps

    def project_world(self, camera: int, objectPoints):
        if camera == 0:
            projected = cv2.projectPoints(
                objectPoints=objectPoints,
                rvec=np.array([[-0.92168101], [0.51675538], [1.16810413]]),
                # flip the x-y axis so that origin starts from camera b0
                tvec=-np.array([[-2.06508224], [2.35084203], [8.02940471]]),
                cameraMatrix=self.cameraMatrix,
                distCoeffs=self.distCoeffs,
            )
        elif camera == 1:
            projected = cv2.projectPoints(
                objectPoints=objectPoints,
                rvec=np.array([[-0.63596615], [-0.9252206], [-1.82702263]]),
                # flip the x-y axis so that origin starts from camera b0
                tvec=-np.array([[2.41941673], [-3.81369888], [15.68241442]]),
                cameraMatrix=self.cameraMatrix,
                distCoeffs=self.distCoeffs,
            )
        return projected[0].squeeze(axis=1)

    def project(self, pos: Tuple[int, int]):
        # Compute ground plane index from world space coordinate
        index = self.get_index(pos)
        pixels = []
        for c, projected in enumerate(self.projections):
            if c == 0:
                offset = np.count_nonzero(self.ground_plane[:index, 0] >= -3)
            elif c == 1:
                offset = np.count_nonzero(self.ground_plane[:index, 0] <= 9)
            else:
                offset = index
            pixels.append(projected[offset])
        return pixels

    def update_positions(self, detections: Dict[int, list]):
        h_detected = {}
        for c, _ in enumerate(self.occupancy_map):
            if len(detections[c]) == 0:
                continue

            # Sort detections to process them in the same order for both cameras
            df = pd.DataFrame(detections[c])
            if df.shape[1] == 6:
                df.columns = ["x0", "y0", "x1", "y1", "label", "confidence"]
            else:
                df.columns = ["x0", "y0", "x1", "y1"]
            df = df.sort_values("y1", ascending=bool(c % 2))

            update = np.zeros(shape=self.grid_size)
            projected = self.projections[c]
            for i in range(df.shape[0]):
                box = df.iloc[i]
                c_x = (box["x0"] + box["x1"]) / 2
                c_y = box["y1"]
                # Use middle of bottom edge of bounding box as detected image point
                detected = np.array([c_x, c_y])
                # Compute euclidean distance with all projected points in image space
                distance = np.linalg.norm(projected - detected, axis=1)
                # Normalize by the softmax of distance inverse to get the confidence score
                # ie. the closer a projected point is to detected point, the higher the score
                normalized = np.exp(-distance)
                normalized /= normalized.sum()
                # Get ground plane indices for all projected points
                if c == 0:
                    index = np.asarray(self.ground_plane[:, 0] >= -3).nonzero()[0]
                elif c == 1:
                    index = np.asarray(self.ground_plane[:, 0] <= 9).nonzero()[0]
                # Unravel by the grid size to get their coordinates in global occupancy map
                to_update = np.unravel_index(index, self.grid_size)
                update[to_update] += normalized
                # Remember top edge of bounding box for height estimation
                g_index = index[normalized.argmax()]
                h_detected[(c, g_index)] = np.array([c_x, box["y0"]])
            self.occupancy_map[c] = update

        num_detection_sets = sum([len(x) > 0 for x in detections.values()])
        if num_detection_sets > 1:
            # --- Estimate height of elephant ---
            targets = self.get_positions_sorted()
            h_range = np.arange(1, 6, 0.01)
            for i in range(max(len(t) for t in targets)):
                for c in range(len(self.projections)):
                    g_index = self.get_index(targets[c][i])
                    g_detected = self.ground_plane[g_index][:-1]
                    # Make candidate elephant height in discrete 1cm intervals
                    h_world = np.array([np.append(g_detected, h) for h in h_range])
                    h_projected = self.project_world(camera=1 - c, objectPoints=h_world)
                    # Compute euclidean distance w.r.t projected points for all cameras
                    g2_index = self.get_index(targets[1 - c][i])
                    h_index = (1 - c, g2_index)
                    if h_index not in h_detected:
                        # Ignore extra detections from either camera
                        continue
                    h_distance = np.linalg.norm(h_projected - h_detected[h_index], axis=1)
                    height = 1 + h_distance.argmin() / 100
                    self.target_height[c][i] = height
            # # Take the average height detected in both cameras
            # print("Diff: ", np.diff(self.target_height, axis=0))
            # print("Height: ", np.mean(self.target_height, axis=0))

    @classmethod
    def align(cls, camera_targets):
        # Use camera with most number of detections as reference
        r_index = np.argmax([len(c) for c in camera_targets])
        ref = np.array(camera_targets[r_index])
        result = []
        for c, targets in enumerate(camera_targets):
            if c == r_index:
                result.append(ref)
                continue
            best = (1e9, [ref])
            # Try all combinations to find the best order that minimises global distance
            for indices in combinations([i for i in range(len(ref))], len(targets)):
                ordered = np.copy(ref)
                ordered[np.array(indices)] = targets
                distance = np.linalg.norm(ordered - ref, axis=1).sum()
                if distance < best[0]:
                    best = (distance, ordered)
            result.append(best[1])
        return result

    def get_positions_sorted(self):
        # Take all high confidence cells from each camera's occupancy map
        camera_targets = self.filter_positions()
        num_target_sets = sum([len(x) > 0 for x in camera_targets])
        if num_target_sets == 1:
            ref_idx = np.argmax([len(x) for x in camera_targets])
            camera_targets = [camera_targets[ref_idx]] * len(self.occupancy_map)
        return self.align(camera_targets)

    def get_positions(self) -> Tuple[int, int]:
        pos = self.get_positions_sorted()
        # Average detections from all cameras for global occupancy map
        return np.mean(pos, axis=0, dtype=int)

    def get_index(self, pos):
        return pos[1] + pos[0] * self.grid_size[1]

    def filter_positions(self):
        return [np.argwhere(grid > self.threshold) for grid in self.occupancy_map]

    def get_likelihood(self, camera: int, pos: Tuple[int, int]) -> float:
        return self.occupancy_map[camera][pos]


def set_aspect_equal(ax, X, Y, Z):
    # Create cubic bounding box to simulate equal aspect ratio
    # https://stackoverflow.com/a/13701747
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], "w")


def plot_3d(rec):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # ax.scatter(
    #     rec.ground_plane[:, 0],
    #     rec.ground_plane[:, 1],
    #     rec.occupancy_map[0].flatten(),
    #     marker="x",
    # )
    X, Y = np.meshgrid(
        np.arange(rec.grid_size[1]) / rec.density,
        np.arange(rec.grid_size[0]) / rec.density,
    )
    # Z = rec.occupancy_map.sum(axis=0)
    Z = rec.occupancy_map[0]
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    # surf = ax.plot_surface(X, Y, rec.occupancy_map[1], cmap=cm.coolwarm)

    set_aspect_equal(ax, X, Y, Z)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()


def main():
    store = S3PredictionStore(
        bucket="my-bucket,
        credentials=AWSCredentials(
            region_name="ap-southeast-1",
        ),
    )
    df = store.load_predictions(prefix="models/predictions/2020-12-01")
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz=None)
    index = pd.Index(df["timestamp"]).unique()

    target = datetime(year=2020, month=12, day=1, hour=0, minute=0, second=0)
    timestamp = index[index.get_loc(target, method="nearest")]
    by_time = df.loc[df["timestamp"] == timestamp]

    rec = Stereo(num_targets=4)
    rec.update_positions(by_time)

    pos = rec.get_positions()
    print(pos)
    print(
        "(X, Y, Z): ",
        [tuple(rec.ground_plane[p[1] + p[0] * rec.grid_size[1]].astype(float)) for p in pos],
    )
    # likelihood = [rec.get_likelihood(i, p) for i, p in enumerate(pos)]
    # print(likelihood)
    # proj_pos = [rec.project(p) for i, p in enumerate(pos) if likelihood[i] > 0.01]
    proj_pos = [rec.project(p) for p in pos]
    print(proj_pos)

    mat = []
    # Stereo frames synchronized at the same timestamp
    images = [
        cv2.imread("calibrate/b0_frame_1.png"),
        cv2.imread("calibrate/b1_frame_1.png"),
    ]
    for c, img in enumerate(images):
        by_cam = by_time[by_time["cam"] == c]
        for i in range(by_cam.shape[0]):
            box = by_cam.iloc[i]
            # img = cv2.circle(
            #     img=img,
            #     center=((box["x0"] + box["x1"]) // 2, box["y1"]),
            #     radius=3,
            #     color=(0, 255, 0),
            #     thickness=3,
            # )
            img = cv2.rectangle(
                img=img,
                pt1=(box["x0"], box["y0"]),
                pt2=(box["x1"], box["y1"]),
                color=(0, 255, 0),
                thickness=3,
            )

        # Draw projected ground plane
        for p in rec.projections[c]:
            if p[0] < 0 or p[0] >= img.shape[1] or p[1] < 0 or p[1] >= img.shape[0]:
                continue
            img = cv2.circle(
                img=img,
                center=tuple(p.astype(int)),
                radius=3,
                color=(255, 0, 0),
                thickness=3,
            )

        # Plot projected centroid
        for p in proj_pos:
            img = cv2.circle(
                img=img,
                center=tuple(p[c].astype(int)),
                radius=3,
                color=(0, 0, 255),
                thickness=3,
            )

        # Plot height estimate
        g_xyz = np.zeros(shape=(len(pos), 3))
        for i, p in enumerate(pos):
            g_xyz[i] = rec.ground_plane[p[1] + p[0] * rec.grid_size[1]]
            g_xyz[i][-1] = rec.target_height[c][i]
        for p in rec.project_world(c, g_xyz):
            img = cv2.circle(
                img=img,
                center=tuple(p.astype(int)),
                radius=3,
                color=(0, 255, 255),
                thickness=3,
            )

        mat.append(img)

    frame = np.concatenate(mat, axis=1)
    frame = cv2.resize(src=frame, dsize=(2000, 750))
    cv2.imshow("frame", frame)
    # cv2.waitKey(0)

    plot_3d(rec)


if __name__ == "__main__":
    main()
