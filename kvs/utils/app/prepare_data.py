"""
Utility functions to compute data for the dashboard.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from utils.app.s3 import s3_read, s3_read_parquet
from utils.serve.daily_stats import filter_neha, form_timeseries, compute_durations
from utils.serve.general import generate_filename, set_sgtz

BUCKET_NAME = "neha-wrs-basisai"
PTRACKS_DIR = "bev_tracks_clean"
STATS_DIR = "daily_stats"


@st.cache
def load_tracks(start_time, end_time):
    """Load tracks from S3."""
    time_range = pd.date_range(start_time, end_time, freq="1H", closed="left")

    dfs = list()
    for t in time_range:
        df = s3_read_parquet(BUCKET_NAME, f"{PTRACKS_DIR}/{generate_filename(t)}.gz.parquet")
        if df is not None:
            dfs.append(df)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        return df
    return


@st.cache
def ts_make_heatmap(df, datestr):
    """Output dataframe in the form of ["timestamp", "value", "mode", "Time", "Date"]."""
    def _get_mode(x):
        if x["Lying down"] == 0 and x["Standing"] == 0:
            return ["Not visible", 0]
        elif x["Lying down"] >= x["Standing"]:
            return ["Lying down", -x["Lying down"]]
        return ["Standing", x["Standing"]]

    source = df[["Lying down", "Standing"]].fillna(0).resample("15T").mean()
    modes = np.array(source.apply(_get_mode, axis=1).values.tolist())
    source["mode"] = modes[:, 0]
    source["value"] = modes[:, 1]
    source = source[["value", "mode"]].reset_index()
    source["Time"] = source["timestamp"].apply(lambda x: x.strftime("%H:%M"))
    source["variable"] = datestr
    return source


@st.cache
def ts_1col_make_heatmap(df, datestr):
    """Output dataframe in the form of ["timestamp", "value", "Time", "Date"]."""
    col = df.columns[0]
    source = df.fillna(0).resample("15T").mean().reset_index()
    source.rename(columns={col: "value"}, inplace=True)
    source["Time"] = source["timestamp"].apply(lambda x: x.strftime("%H:%M"))
    source["variable"] = datestr
    return source


@st.cache
def ts_make_linechart(df, cols=["Lying down", "Standing"]):
    """Output dataframe in the form of ["timestamp", "variable", "value"]."""
    source = df.reset_index()
    source["timestamp"] = source["timestamp"].dt.tz_convert("UTC")
    source = pd.melt(source, id_vars=["timestamp"], value_vars=cols)
    return source


def _set_order(x):
    if x == "Active":
        return 0
    if x == "Inactive":
        return 1
    if x == "Not visible":
        return 4
    if x == "Standing":
        return 3
    if x == "Lying down":
        return 2
    return -1


@st.cache
def get_daily_stats(start_time, end_time):
    """Compute dataframe for daily_stats page."""
    # end_time = start_time + timedelta(days=1)
    postproc_df = load_tracks(start_time, end_time)
    if postproc_df is None:
        return

    # Filter Neha only
    df = filter_neha(postproc_df)
    ts = form_timeseries(df, start_time, end_time)
    
    # active_hrs, inactive_hrs, notvisible, standing_hrs, lying_hrs
    act_hrs = [x / 3600 for x in compute_durations(ts)]
    rows = [["Neha", *act_hrs]]

    value_vars = ["Active", "Inactive", "Not visible", "Standing", "Lying down"]
    df = pd.DataFrame(rows, columns=["ID"] + value_vars)

    # # add whole herd average
    # df.loc[len(df)] = ["herd"] + df[value_vars].mean(axis=0).values.tolist()

    # melt dataframe
    df = pd.melt(df, id_vars=["ID"], value_vars=value_vars,
                 var_name="Activity", value_name="Hours")
    df["order"] = df["Activity"].apply(_set_order)
    return df


@st.cache
def get_neha_track(start_time, end_time):
    """
    Load and filter Neha's trajectory.
    Output timeseries with columns ["motion", "Standing", "Lying down"]
    and activity hours.
    """
    postproc_df = load_tracks(start_time, end_time)
    if postproc_df is None:
        return
    
    # Filter Neha only
    df = filter_neha(postproc_df)
    ts = form_timeseries(df, start_time, end_time)

    subset_ts = ts.query("timestamp >= @start_time and timestamp < @end_time")
    act_durations = compute_durations(subset_ts)
    return subset_ts, act_durations


@st.cache
def get_history(date_range, time_range):
    """
    Load and filter Neha's trajectories for the given date and time ranges.
    Output list of timeseries with columns ["motion", "Standing", "Lying down"]
    and activity hours.
    """
    start_times, end_times = [
        pd.date_range(
            datetime.combine(date_range[0], t),
            datetime.combine(date_range[1], t),
        ) for t in time_range
    ]
    if end_times[0] <= start_times[0]:
        end_times += timedelta(days=1)

    dates_list = list()
    ts_list = list()
    hrs_list = list()
    for start_time, end_time in zip(start_times, end_times):
        results = get_neha_track(set_sgtz(start_time), set_sgtz(end_time))
        if results is None:
            continue

        dates_list.append(start_time.strftime("%Y-%m-%d"))
        ts_list.append(results[0])
        hrs_list.append(results[1])
    
    if len(ts_list) > 0 and len(hrs_list) > 0:
        return dates_list, ts_list, hrs_list
    return


def _get_ts_heatmap(ts, datestr, cols):
    if len(cols) == 2 and "Standing" in cols and "Lying down" in cols:
        return ts_make_heatmap(ts, datestr)
    if len(cols) == 1:
        return ts_1col_make_heatmap(ts[[cols[0]]], datestr)
    raise Exception("Not implemented")


@st.cache
def concat_ts(ts_list, dates_list, cols):
    """Concatenate timeseries rowwise."""
    all_ts = list()
    for ts, datestr in zip(ts_list, dates_list):
        all_ts.append(_get_ts_heatmap(ts, datestr, cols))
    return pd.concat(all_ts, ignore_index=True)


@st.cache
def load_daily_activities(start_time, end_time, start_date=None, end_date=None):
    """Load daily statistics data."""
    key = f"{STATS_DIR}/daily_activities.csv"
    df = pd.read_csv(s3_read(BUCKET_NAME, key))

    df = df.query("Start_time == @start_time & End_time == @end_time")
    if start_date is not None:
        df = df.query("Date >= @start_date")
    if end_date is not None:
        df = df.query("Date <= @end_date")

    df = df.reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df.set_index("Date", inplace=True)
    return df
