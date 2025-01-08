"""
Utility functions for daily stats.
"""
import numpy as np
import pandas as pd


def filter_neha(postproc_df, threshold=0.3):
    """Filter Neha trajectory."""
    # TODO: does not work well
    # avg_neha = postproc_df.groupby("object_id")["neha_pred"].mean()
    # neha_ids = avg_neha[avg_neha > threshold].index
    # neha_df = postproc_df.query("object_id in @neha_ids")

    neha_df = postproc_df.query("neha_pred == 1")
    return neha_df


def _resample(df, rule):
    ts_df = (
        df
        .set_index("timestamp")
        .resample(rule)
        .mean()
    )
    return ts_df


def form_timeseries(df, start_time, end_time, rule="1S"):
    """
    From input dataframe in the form of
    ["object_id", "timestamp", "cam", "label", "conf", "cx", "cy"],
    create dataframe in the form of
    ["object_id", "timestamp", "motion", "Standing", "Lying down"].
    """
    ts_df = pd.DataFrame(index=pd.date_range(start_time, end_time, freq=rule))
    ts_df.index.name = "timestamp"

    motion_df = _resample(df[["timestamp", "cx", "cy"]], "1S")
    motion_df["motion"] = motion_df["cx"].diff() + motion_df["cy"].diff()
    ts_df = ts_df.join(motion_df[["motion"]])

    labels = [0, 1]
    label_names = ["raw_standing", "raw_lying"]
    for label, name in zip(labels, label_names):
        _label_df = df.query(f"label == {label}")[["timestamp", "conf"]].copy()
        if len(_label_df) > 0:
            _label_df = _resample(_label_df, rule)
            _label_df.columns = [name]
            ts_df = ts_df.join(_label_df)
        else:
            ts_df[name] = None

    ts_df["standing_tmp"] = ts_df["raw_standing"].fillna(0)
    ts_df = ts_df.resample("1T").mean()

    ts_df["mode"] = np.where(
        ts_df["raw_lying"] > ts_df["standing_tmp"], "Lying down", "Standing")
    ts_df["Standing"] = np.where(
        ts_df["mode"] == "Standing", ts_df["raw_standing"], np.NaN)
    ts_df["Lying down"] = np.where(
        ts_df["mode"] == "Lying down", ts_df["raw_lying"], np.NaN)
    ts_df["mode"] = np.where(
        ts_df["Standing"].isnull() & ts_df["Lying down"].isnull(), None, ts_df["mode"])
    return ts_df


def compute_durations(df, moving_thres=0.1, mode_thres=0.7):
    """Compute activity durations (in secs) from timeseries.
    
    - not visible
    - active
      - moving
    - inactive
      - standing
      - lying down
    """
    factor = (df.index[1] - df.index[0]).seconds
    active = ((df["motion"] > moving_thres) & (df["Standing"] > mode_thres)).sum() * factor
    standing = ((df["motion"] < moving_thres) & (df["Standing"] > mode_thres)).sum() * factor
    lying = (df["Lying down"] > mode_thres).sum() * factor
    inactive = standing + lying
    notvisible = (df["Standing"].isnull() & df["Lying down"].isnull()).sum() * factor
    return active, inactive, notvisible, standing, lying
