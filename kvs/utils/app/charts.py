"""
Utility functions for charts used in the dashboard.
"""

import altair as alt
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def plot_barchart(source):
    """Custom bar chart."""
    _domain = ["Active", "Inactive", "Lying down", "Standing", "Not visible"]
    _range = ["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4", "black"]
    bar = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            x=alt.X("Activity:N", sort=alt.SortField("order")),
            y="Hours:Q",
            color=alt.Color(
                "Activity:N",
                scale=alt.Scale(domain=_domain, range=_range),
                legend=None,
            ),
            column=alt.Column("ID:N", title=""),
            tooltip=[
                alt.Tooltip("Activity"),
                alt.Tooltip("Hours", format=".1f"),
            ],
        )
    )
    return bar


def plot_heatmap(source, ytitle="", cmap="redblue", domain=(-1, 1), sort=None):
    """Custom heatmap chart."""
    tooltip = [
        alt.Tooltip("Time:O", title="Time"),
        # alt.Tooltip("value:Q", title="score"),
    ]
    if "mode" in source.columns:
        tooltip.append(alt.Tooltip("mode:N", title="Mode"))

    bar = (
        alt.Chart(source)
        .mark_rect()
        .encode(
            x=alt.X("Time:O", title="Time", sort=sort, axis=alt.Axis(labels=False)),
            y=alt.Y("variable:O", title=ytitle),
            color=alt.Color("value:Q", scale=alt.Scale(scheme=cmap, domain=domain), legend=None),
            tooltip=tooltip,
        )
    )
    return bar


def plot_linechart(
    source,
    xtitle="Time",
    xformat=None,
    ytitle="",
    yscale=(0, 1),
    tformat="%Y-%m-%d %H:%M",
    crange=None,
    title="",
):
    """Custom line chart."""
    xargs = {"title": xtitle}
    if xformat:
        xargs["axis"] = alt.Axis(format=xformat)

    yargs = {"title": ytitle}
    if yscale:
        yargs["scale"] = alt.Scale(domain=yscale)

    line = (
        alt.Chart(source)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", **xargs),
            y=alt.Y("value:Q", **yargs),
            tooltip=[
                alt.Tooltip("timestamp:T", title=xtitle, format=tformat),
                alt.Tooltip("value:Q", title=ytitle),
            ],
        )
        .properties(title=title)
    )

    if "variable" in source.columns:
        cargs = {"title": None}  # , "legend": alt.Legend(orient="bottom")
        if crange:
            cargs["scale"] = alt.Scale(range=crange)
        line = line.encode(color=alt.Color("variable:N", **cargs))
    return line


def ring_viz(ax, data1, labels1, colors1, data2=None, labels2=None, colors2=None, title=None):
    ax.axis("equal")
    width = 0.25

    pie1, _ = ax.pie(data1[::-1], radius=1, colors=colors1[::-1], startangle=90)
    plt.setp(pie1, width=width, edgecolor="white")

    # setting up the legend
    bars = list()
    for label, color in zip(labels1, colors1):
        bars.append(
            mlines.Line2D(
                [], [], color=color, marker="s", linestyle="None", markersize=10, label=label
            )
        )

    if data2 is not None:
        pie2, _ = ax.pie(data2[::-1], radius=1 - width, colors=colors2[::-1], startangle=90)
        plt.setp(pie2, width=width, edgecolor="white")

        for label, color in zip(labels2, colors2):
            bars.append(
                mlines.Line2D(
                    [], [], color=color, marker="s", linestyle="None", markersize=10, label=label
                )
            )

    ax.legend(handles=bars, prop={"size": 8}, loc="center", frameon=False)

    if title is not None:
        ax.set_title(title)


def plot_rings(active, inactive, notvisible, standing, lying):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)

    ring_viz(
        axs[0],
        [notvisible, active, inactive],
        # ["Not visible", "Active", "Inactive"],
        [
            f"{notvisible / 3600:.0f} hrs not visible",
            f"{active / 3600:.0f} hrs active",
            f"{inactive / 3600:.0f} hrs inactive",
        ],
        ["black", "#2ca02c", "#ff7f0e"],
    )
    ring_viz(
        axs[1],
        [notvisible, active, lying, standing],
        # ["Not visible", "Moving", "Lying down", "Standing"],
        [
            f"{notvisible / 3600:.0f} hrs not visible",
            f"{active / 3600:.0f} hrs moving",
            f"{lying / 3600:.0f} hrs lying down",
            f"{standing / 3600:.0f} hrs standing",
        ],
        ["black", "#2ca02c", "#d62728", "#1f77b4"],
    )
    return fig
