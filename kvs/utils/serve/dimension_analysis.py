"""
Neha identification.
"""
from typing import Dict, Tuple
from functools import reduce
import numpy as np
import pandas as pd
from utils.serve.general_bev import camera_to_world, world_to_camera


def estimate_height(pos: Tuple, top_edge: Tuple, camera: Dict):
    """Estimate height using pos and top_edge."""
    # Make candidate elephant height in discrete 1cm intervals
    start, stop, step = 0, 6, 0.01     
    h_range = np.arange(start, stop, step)
    n = len(h_range)
    offset = camera["offset"][:2]
    # candidate heights to try
    h_world = np.hstack(
        [np.tile(np.array(pos) - offset, (n, 1)), h_range.reshape(-1, 1)])
    # Project all candidates into the camera's image space
    h_projected = world_to_camera(camera=camera, objectPoints=h_world)
    # Compute euclidean distance w.r.t all projected points
    h_distance = np.linalg.norm(h_projected - top_edge, axis=1)
    # Index of the closest point is mapped back to actual height
    height = start + h_distance.argmin() * step
    return height


def height_estimation(df: pd.DataFrame, cameras: Dict):
    """Estimate height."""
    df["height"] = df.apply(lambda x: estimate_height(
        (x["cx"], x["cy"]), x["top_edge"], cameras[x["cam"]]), axis=1)
    return df


def body_length_estimation(df: pd.DataFrame, cameras: Dict):
    """
    Estimate body length using the long side of bbox, calculate body length for
    three pairs of points for each bbox, and select the minumum value.
    """
    df1 = df.copy()
    xory = df1.eval("(x1 - x0) > (y1 - y0)").astype(int)
    # # if (x1 - x0) > (y1 - y0), use x dimension to estimate body length,
    # otherwise, use y dimension to estimate body length
    df1["l1"] = list(zip(df1.x0, df1.y0))
    df1["l2"] = list(zip(
        np.where(xory == 1, df1.x0, df1.x1),
        np.where(xory == 1, df1.y1, df1.y0),
    ))
    df1["l3"] = list(zip(
        np.where(xory == 1, df1.x0, (df1.x0 + df1.x1) / 2),
        np.where(xory == 1, (df1.y0 + df1.y1) / 2, df1.y0),
    ))
    df1["r1"] = list(zip(
        np.where(xory == 1, df1.x1, df1.x0),
        np.where(xory == 1, df1.y0, df1.y1),
    ))
    df1["r2"] = list(zip(df1.x1, df1.y1))
    df1["r3"] = list(zip(
        np.where(xory == 1, df1.x1, (df1.x0 + df1.x1) / 2),
        np.where(xory == 1, (df1.y0 + df1.y1) / 2, df1.y1),
    ))

    for c in ["dist1", "dist2", "dist3"]:
        df1[c] = None
    for cam in df1["cam"].unique():
        indices = (df1["cam"] == cam)
        for c1, c2, c3 in zip(["l1", "l2", "l3"], ["r1", "r2", "r3"],
                              ["dist1", "dist2", "dist3"]):
            c1 = np.array(df1[c1].loc[indices].values.tolist())
            c1_world = camera_to_world(cameras[cam], c1)[:, :2]

            c2 = np.array(df1[c2].loc[indices].values.tolist())
            c2_world = camera_to_world(cameras[cam], c2)[:, :2]

            df1.loc[indices, c3] = np.linalg.norm(c1_world - c2_world, axis=1)

    df1["body_length"] = df1[["dist1", "dist2", "dist3"]].min(axis=1)
    df1.drop(columns=[
        "l1", "l2", "l3", "r1", "r2", "r3", "dist1", "dist2", "dist3"], inplace=True)
    return df1


def dimension_analysis(
    tracks_df: pd.DataFrame,
    feature="height",  # body_length
    feature_range=(1.5, 3.5),  # tried (2, 4.5) for body length estimation
    window="5s",  # in time frequency string
    min_periods=1,
    average_each=True,
    exclude_one_bbox=False,
):
    """
    Estimate height / body length and identify object_id with minimum height / body length
    at each timestamp.
    """
    _cols = [
        'timestamp', 'object_id', 'cx', 'cy', 'label', 'conf', 
        feature, f'rolling_{feature}', f'min_{feature}', f'{feature}_pred',
    ]

    df = tracks_df.copy()
    # Filter data by data range
    if feature_range is not None:
        df = df[df[feature].between(*feature_range)]
        if df.empty:
            return pd.DataFrame(columns=_cols)

    # Keep only object_ids with more than one bboxes
    if exclude_one_bbox:
        bbox_num = (
            df
            .groupby(["timestamp", "object_id"])[feature]
            .transform("count")
        )
        df = df[bbox_num > 1]

    # Compute average height / body length estimates of each object_id at each timestamp
    if average_each:
        df[feature] = (
            df
            .groupby(["timestamp", "object_id"])[feature]
            .transform("mean")
        )

    # Take label with the highest conf. Take first row if multiple rows have the same max_conf
    df["max_conf"] = (
        df
        .groupby(["timestamp", "object_id"])["conf"]
        .transform("max")
    )
    df1 = (
        df
        .query("conf == max_conf")[["timestamp", "object_id", "cx", "cy", "label", "conf", feature]]
        .groupby(["timestamp", "object_id"])
        .first()
        .reset_index()
    )
    
    # Compute rolling mean height / body length in the last 'window' secs for each object_id
    tmp = (
        df1
        .sort_values("timestamp")
        .set_index("timestamp")
        .groupby("object_id")[feature]
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(name=f"rolling_{feature}")
    )
    df1 = df1.merge(tmp, how="left", on=["timestamp", "object_id"])

    # current hypothesis: object_id with minimum height / body length at each timestamp is Neha.
    df1[f"min_{feature}"] = df1.groupby(["timestamp"])[f"rolling_{feature}"].transform("min")
    df1[f"{feature}_pred"] = (df1[f"rolling_{feature}"] == df1[f"min_{feature}"]).astype(int)
    return df1[_cols]


def neha_predict(df: pd.DataFrame):
    """Predict Neha.
    Current hypothesis: object_id with minimum height / body length at each timestamp is Neha.
    """
    df.sort_values(["timestamp", "object_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # neha identification using height and body_length estimates
    height_df = dimension_analysis(df, feature="height", feature_range=(1.5, 3.5))
    length_df = dimension_analysis(df, feature="body_length", feature_range=(2, 4.5))

    cols = ["timestamp", "object_id"]
    tmp = df[cols].drop_duplicates().reset_index(drop=True)
    df_ls = [tmp, height_df, length_df.drop(columns=["cx", "cy", "label", "conf"])]
    pred_df = reduce(lambda df1, df2: pd.merge(df1, df2, how='left', on=cols), df_ls)

    # using height_pred by default, but use body_length_pred when there is elephant lying down
    num_lying_down = pred_df.groupby("timestamp")["label"].transform("sum")
    pred_df["neha_pred"] = np.where(
        num_lying_down > 0, pred_df["body_length_pred"], pred_df["height_pred"])
    return pred_df
