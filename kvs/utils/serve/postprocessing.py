"""
Postprocessing.
"""
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


def _round_timestamp(df: pd.DataFrame):
    df["rounded_timestamp"] = df["timestamp"].dt.floor(freq="1T")
    return df


def calibrate_body_length(df: pd.DataFrame, b2h_ratio=1.5):
    """Calibrate body length when elephant's body length to height ratio (b2h_ratio) < 1.5."""
    cols = ["timestamp", "object_id", "x0", "y0", "x1", "y1", "rolling_body_length"]
    df1 = df[cols].copy()
    df1["lenx"] = df1["x1"] - df1["x0"]
    df1["leny"] = df1["y1"] - df1["y0"]
    df1["ratio"] = np.where(
        df1["lenx"] >= df1["leny"],
        df1["lenx"] / df1["leny"],
        df1["leny"] / df1["lenx"]
    )
    df1["ratio"] = df1.groupby(["timestamp", "object_id"]).ratio.transform("mean")

    df2 = (
        df1[["timestamp", "object_id", "ratio", "rolling_body_length"]]
        .drop_duplicates()
        .set_index("timestamp")
    )
    df2['ratio'] = df2.groupby(['object_id'])['ratio'].transform(lambda x: x.rolling('10S').mean())
    df2["calibrated_rolling_body_length"] = np.where(
        df2["ratio"] < b2h_ratio,
        df2["rolling_body_length"] / df2["ratio"] * b2h_ratio,
        df2["rolling_body_length"]
    )
    df2["rounded_body_length"] = df2["calibrated_rolling_body_length"].astype(float).round(1)
    df2.drop(columns=["ratio", "rolling_body_length"], inplace=True)
    return df2.reset_index().drop_duplicates()


def select_hybrid_scenarios(df: pd.DataFrame):
    """Select scenarios where some elephants standing and some lying down."""
    df1 = df.copy()
    df1["conf"] = df1.groupby(["timestamp", "object_id"])["conf"].transform("max")
    df1["label"] = df1.groupby(["timestamp", "object_id"])["label"].transform("max")

    df2 = df1[["timestamp", "object_id", 'label']].drop_duplicates().reset_index(drop=True)
    df2["num_objects"] = df2.groupby("timestamp")["object_id"].transform("count")
    df2["lying_down"] = df2.groupby("timestamp")["label"].transform("sum")
    df2["hybrid"] = df2["lying_down"] / df2["num_objects"]
    timestamp_ls = list(
        df2[df2["hybrid"].between(0, 1, inclusive=False)]
        .timestamp.dt.floor("1T").unique()
    )
    return df1[df1.timestamp.dt.floor("1T").isin(timestamp_ls)].reset_index(drop=True)


def correct_neha_prediction(
    pred_df: pd.DataFrame,
    mode_df: pd.DataFrame,
    datetime_ls: List,
    value_col: str,   # rolling_height / rolling_body_length
    pred_col: str,  # neha_standing / neha_lying_down
    mode: str,  # standing / lying_down
):
    """Correct Neha prediction when some standing some lying down."""
    df = pred_df.sort_values("timestamp").reset_index(drop=True)
    df = _round_timestamp(df)
    # Select timestamps when neha in mode
    tmp = df[df.rounded_timestamp.isin(datetime_ls)].reset_index(drop=True)
    tmp = tmp.merge(mode_df, how="left", on=["timestamp", "object_id"])

    # Identify Neha's object_id when object's height / body_length == minimum height / body_length
    # at each tiimestamp, corrected_col = 0 for other object_ids.
    corrected_col = f"{mode}_corrected"
    tmp["tmp"] = tmp[tmp[pred_col] == 1].groupby("timestamp")[value_col].transform("min")
    tmp[corrected_col] = np.where(tmp[value_col] == tmp["tmp"], 1, np.nan)
    timestamp_ls = list(tmp[tmp[value_col] == tmp["tmp"]]["timestamp"].unique())
    indices = tmp.timestamp.isin(timestamp_ls)
    tmp.loc[indices, corrected_col] = tmp.loc[indices, corrected_col].fillna(0)

    # Correct Neha predictions
    cols = ["timestamp", "object_id", corrected_col]
    df = df.merge(tmp[cols].drop_duplicates().reset_index(drop=True),
                  how='left', on=["timestamp", "object_id"])
    df.loc[df[corrected_col].notna(), "neha_pred"] = df[corrected_col]
    df.drop(columns=["rounded_timestamp", corrected_col], inplace=True)
    return df


def postprocess_hybrid_scenarios(
    merged_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    start_time: datetime,
    height_thresholds: Dict[int, tuple],
    body_length_thresholds: Dict[int, float],
):
    """Postprocessing step to handle cases where some elephants standing and some lying down."""
    # when there's elephant lying down, check height first
    df = merged_df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calibrate body length
    tmp = calibrate_body_length(df, b2h_ratio=1.5)
    df = df.merge(tmp, how="left", on=["timestamp", "object_id"])
    pred_df = pred_df.merge(tmp, how="left", on=["timestamp", "object_id"])

    # Select scenarios where some elephants standing and some lying down
    tmp = select_hybrid_scenarios(df)
    cols = ["timestamp", "object_id", "label", "cam", "rolling_height", "rounded_body_length"]
    tmp = tmp[cols].drop_duplicates().reset_index(drop=True)

    # Check if any elephant standing is Neha
    # Criteria: rolling_height within height range of Neha
    tmp["neha_standing"] = 0
    for cid in tmp["cam"].unique():
        indices = tmp[
            (tmp["cam"] == cid)
            & (tmp["label"] == 0)
            & (tmp["rolling_height"].between(*height_thresholds[cid]))
        ].index
        tmp.loc[indices, "neha_standing"] = 1
    # standing_df for correcting neha prediction in later stage
    standing_df = tmp[["timestamp", "object_id", "neha_standing"]].drop_duplicates().reset_index(drop=True)
    tmp["neha_standing"] = tmp.groupby("timestamp")["neha_standing"].transform("sum").clip(upper=1)
    tmp = _round_timestamp(tmp)

    tmp1 = (
        tmp[["timestamp", "neha_standing"]]
        .drop_duplicates()
        .set_index("timestamp")
    )
    tmp1["rolling_neha_standing"] = tmp1["neha_standing"].rolling("5T", min_periods=50).mean()
    tmp1.reset_index(inplace=True)

    tmp1 = tmp1.query("timestamp >= @start_time & (rolling_neha_standing >= 0.7)").reset_index(drop=True)
    tmp1 = _round_timestamp(tmp1)
    standing_datetime = list(tmp1["rounded_timestamp"].unique())
    lydown_datetime = list(set(tmp.rounded_timestamp.unique()) - set(standing_datetime))

    # If no elephant standing is Neha, check if any elephant lying down is Neha
    # Criteria: rounded_body_length <= upper bound of Neha's body length
    tmp2 = tmp[(tmp["rounded_timestamp"].isin(lydown_datetime)) & (tmp.label == 1)].reset_index(drop=True)
    tmp2["neha_lying_down"] = 0
    for cid in tmp2["cam"].unique():
        index = tmp2[(tmp2["cam"] == cid)
                     & (tmp2["rounded_body_length"] <= body_length_thresholds[cid])].index
        tmp2.loc[index, "neha_lying_down"] = 1
    # lydown_df for correcting Neha prediction in later stage
    lydown_df = tmp2[["timestamp", "object_id", "neha_lying_down"]].drop_duplicates().reset_index(drop=True)
    tmp3 = tmp2.groupby("timestamp")["neha_lying_down"].sum().clip(upper=1)
    tmp3 = tmp3.rolling("5T", min_periods=50).mean()

    fuzzy_time = list(tmp3[tmp3 < 0.2].index.floor('1T').unique())
    lydown_datetime = list(set(tmp.rounded_timestamp.unique()) - set(standing_datetime) - set(fuzzy_time))

    pred_df = correct_neha_prediction(
        pred_df,
        standing_df,
        standing_datetime,
        value_col="rolling_height",
        pred_col="neha_standing",  # or "neha_lying_down" as defined in postprocess_hybrid_scenarios
        mode="standing"
    )

    pred_df = correct_neha_prediction(
        pred_df,
        lydown_df,
        lydown_datetime,
        value_col="rolling_body_length",
        pred_col="neha_lying_down",
        mode="lying_down",
    )
    return pred_df


def post_processing(
    pred_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    start_time: datetime,
    num_objects: int,
    height_thresholds: Dict[int, Tuple],
):
    """
    Postprocessing step to handle cases where < num_objects elephants present
    and all elephants are standing.
    """
    # # Get previous 5 mins tracks from last hour to calculate the rolling average
    # start_time, tracks_df = get_tracks(tracks_df, postproc_df)

    # Calculate rolling average of objects # and lying down
    # apply this postprocessing step to cases where all detected elephants are standing
    # and # of elephants < num_objects
    tracks_df = merged_df.sort_values("timestamp").copy()

    tracks_df["num_objects"] = tracks_df.groupby("timestamp")["object_id"].transform("nunique")
    tracks_df["lying_down"] = tracks_df.groupby("timestamp")["label"].transform("sum")
    tmp = (
        tracks_df[["timestamp", "num_objects", "lying_down"]]
        .drop_duplicates()
        .set_index("timestamp")
    )
    tmp.loc[tmp["lying_down"] > 0, "lying_down"] = 1
    tmp["rolling_obj_num"] = tmp["num_objects"].rolling("5T").mean()
    tmp["rolling_lying_down"] = tmp["lying_down"].rolling("1T").mean()
    tmp.reset_index(inplace=True)
    df = tracks_df.merge(
        tmp[["timestamp", "rolling_obj_num", "rolling_lying_down"]], how="left", on="timestamp")

    # Get cases where <= num_objects present and all detected elephants are standing
    tmp = df[
        (df.rolling_obj_num <= num_objects)
        & ((df["lying_down"] == 0) | (df["rolling_lying_down"] < 0.01))
    ].reset_index(drop=True)

    # We say neha is present if there is elephant with height within height_thresholds
    # at each timestamp. This step can filter out most wrong human being detections, but not all
    tmp["present"] = 0
    for cid in tmp["cam"].unique():
        indices = ((tmp["cam"] == cid) & (tmp["rolling_height"].between(*height_thresholds[cid])))
        tmp.loc[indices, "present"] = 1
    tmp["present"] = tmp.groupby("timestamp")["present"].transform("sum").clip(upper=1)

    # Calculate rolling confidence for filtering out wrong human being detections
    # who usually has low conf < 0.8
    tmp.set_index("timestamp", inplace=True)
    # tmp.index = tmp.index.tz_convert("Asia/Singapore")
    tmp["rolling_conf"] = tmp.groupby("object_id")["conf"].transform(
        lambda x: x.rolling("1T", min_periods=100).mean())
    tmp.reset_index(inplace=True)
    subset = tmp.query("rolling_conf > 0.8").reset_index(drop=True)

    # list of datetime to correct neha_pred
    tmp = _round_timestamp(tmp)
    subset = _round_timestamp(subset)
    datetime_ls = list(set(tmp.rounded_timestamp.unique()) - set(subset.rounded_timestamp.unique()))

    # Calculate rolling presence of neha, if rolling_present < 0.8 (can adjust),
    # we think it's not neha since its presence is not sustained
    subset = (
        subset[["timestamp", "present"]]
        .drop_duplicates()
        .set_index("timestamp")
    )
    subset["rolling_present"] = subset["present"].rolling("5T", min_periods=100).mean()
    subset.reset_index(inplace=True)

    # Correct neha_pred when neha is absent, round timestamp to minute
    subset["rolling_present"] = subset["rolling_present"].fillna(0)
    tmp = subset.query("timestamp >= @start_time & rolling_present < 0.8").reset_index(drop=True)
    # tmp = subset.query(
    #     "timestamp >= @start_time & (rolling_present < 0.8 | rolling_present.isna())"
    # ).reset_index(drop=True)
    tmp = _round_timestamp(tmp)
    datetime_ls += list(tmp["rounded_timestamp"].unique())

    postproc_df = _round_timestamp(pred_df)
    postproc_df.loc[postproc_df["rounded_timestamp"].isin(datetime_ls), "neha_pred"] = 0
    postproc_df["neha_pred"] = postproc_df["neha_pred"].fillna(0)
    postproc_df.drop(columns="rounded_timestamp", inplace=True)
    return postproc_df
