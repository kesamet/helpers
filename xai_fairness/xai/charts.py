import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pdpbox import pdp


@st.cache(allow_output_mutation=True)
def compute_pdp_isolate(model, dataset, model_features, feature):
    pdp_isolate_out = pdp.pdp_isolate(
        model=model,
        dataset=dataset,
        model_features=model_features,
        feature=feature,
        num_grid_points=15,
    )
    return pdp_isolate_out


def pdp_chart(pdp_isolate_out, feature_name):
    """Plot pdp charts."""
    source = pd.DataFrame(
        {
            "feature": pdp_isolate_out.feature_grids,
            "value": pdp_isolate_out.pdp,
        }
    )

    if pdp_isolate_out.feature_type == "numeric":
        base = alt.Chart(source).encode(
            x=alt.X("feature", title=feature_name),
            y=alt.Y("value", title=""),
            tooltip=["feature", "value"],
        )
        line = base.mark_line()
        scatter = base.mark_circle(size=60)
        chart = line + scatter
    else:
        source["feature"] = source["feature"].astype(str)
        chart = (
            alt.Chart(source)
            .mark_bar()
            .encode(
                x=alt.X("value", title=""),
                y=alt.Y("feature", title=feature_name, sort="-x"),
                tooltip=["feature", "value"],
            )
        )
    return chart


@st.cache(allow_output_mutation=True)
def compute_pdp_interact(model, dataset, model_features, features):
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=dataset,
        model_features=model_features,
        features=features,
    )
    return pdp_interact_out


def pdp_heatmap(pdp_interact_out, feature_names):
    """Plot pdp heatmap."""
    source = pdp_interact_out.pdp

    for i in [0, 1]:
        if pdp_interact_out.feature_types[i] == "onehot":
            value_vars = pdp_interact_out.feature_grids[i]
            id_vars = list(set(source.columns) - set(value_vars))
            source = pd.melt(
                source,
                value_vars=value_vars,
                id_vars=id_vars,
                var_name=feature_names[i],
            )
            source = source[source["value"] == 1].drop(columns=["value"])

        elif pdp_interact_out.feature_types[i] == "binary":
            source[feature_names[i]] = source[feature_names[i]].astype(str)

    chart = (
        alt.Chart(source)
        .mark_rect()
        .encode(
            x=feature_names[0],
            y=feature_names[1],
            color="preds",
            tooltip=feature_names + ["preds"],
        )
    )
    return chart


def _convert_name(ind, feature_names):
    """Get index of feature name if it is given."""
    if isinstance(ind, str):
        return np.where(np.array(feature_names) == ind)[0][0]
    return ind


def make_source_dp(shap_values, features, feature_names, feature):
    ind = _convert_name(feature, feature_names)

    # randomize the ordering so plotting overlaps are not related to
    # data ordering
    oinds = np.arange(shap_values.shape[0])
    np.random.shuffle(oinds)

    return pd.DataFrame(
        {
            feature: features[oinds, ind],
            "value": shap_values[oinds, ind],
        }
    )


def _is_numeric(series, max_unique=16):
    """Flag if series is numeric."""
    if len(set(series.values[:3000])) > max_unique:
        return True
    return False


def dependence_chart(source, feat_col, val_col="value"):
    if _is_numeric(source[feat_col]):
        scatterplot = (
            alt.Chart(source)
            .mark_circle(size=8)
            .encode(
                x=alt.X(f"{feat_col}:Q"),
                y=alt.Y(f"{val_col}:Q", title="SHAP value"),
            )
        )
        return scatterplot

    stripplot = (
        alt.Chart(source, width=40)
        .mark_circle(size=8)
        .encode(
            x=alt.X(
                "jitter:Q",
                title=None,
                axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
                scale=alt.Scale(),
            ),
            y=alt.Y(f"{val_col}:Q", title="SHAP value"),
            color=alt.Color(f"{feat_col}:N", legend=None),
            column=alt.Column(
                f"{feat_col}:N",
                header=alt.Header(
                    labelAngle=-90,
                    titleOrient="top",
                    labelOrient="bottom",
                    labelAlign="right",
                    labelPadding=3,
                ),
            ),
        )
        .transform_calculate(
            # Generate Gaussian jitter with a Box-Muller transform
            jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
        )
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )
    return stripplot


# XAI for single data point
def make_source_waterfall(instance, base_value, shap_values, max_display=10):
    """Prepare dataframe for waterfall chart."""
    df = pd.melt(instance)
    df.columns = ["feature", "feature_value"]
    df["shap_value"] = shap_values

    df["val_"] = df["shap_value"].abs()
    df = df.sort_values("val_", ascending=False)

    df["val_"] = df["shap_value"].values
    remaining = df["shap_value"].iloc[max_display:].sum()
    output_value = df["shap_value"].sum() + base_value

    _df = df.iloc[:max_display]

    df0 = pd.DataFrame(
        {
            "feature": ["Average Model Output"],
            "shap_value": [base_value],
            "val_": [base_value],
        }
    )
    df1 = _df.query("shap_value > 0").sort_values("shap_value", ascending=False).copy()
    df2 = pd.DataFrame(
        {
            "feature": ["Others"],
            "shap_value": [remaining],
            "val_": [remaining],
        }
    )
    df3 = _df.query("shap_value < 0").sort_values("shap_value").copy()
    df4 = pd.DataFrame(
        {
            "feature": ["Individual Observation"],
            "shap_value": [output_value],
            "val_": [0],
        }
    )
    source = pd.concat([df0, df1, df2, df3, df4], axis=0, ignore_index=True)

    source["close"] = source["val_"].cumsum()
    source["open"] = source["close"].shift(1)
    source.loc[len(source) - 1, "open"] = 0
    source["open"].fillna(0, inplace=True)
    return source


# def make_source_waterfall(shap_exp, max_display=10):
#     """Prepare dataframe for waterfall chart."""
#     base_value = shap_exp.base_values
#     df = pd.DataFrame({
#         "feature": shap_exp.feature_names,
#         "feature_value": shap_exp.data,
#         "shap_value": shap_exp.values,
#     })

#     df["abs_val"] = df["shap_value"].abs()
#     df = df.sort_values("abs_val", ascending=True)

#     df["val_"] = df["shap_value"].values
#     remaining = df["shap_value"].iloc[:-max_display].sum()
#     output_value = df["shap_value"].sum() + base_value

#     df0 = pd.DataFrame({
#         "feature": ["Average Model Output"],
#         "shap_value": [base_value],
#         "val_": [base_value],
#     })
#     df2 = pd.DataFrame({
#         "feature": [f"{len(df) - max_display} other features"],
#         "shap_value": [remaining],
#         "val_": [remaining],
#     })
#     df1 = df.iloc[-max_display:]
#     df4 = pd.DataFrame({
#         "feature": ["Individual Observation"],
#         "shap_value": [output_value],
#         "val_": [0],
#     })
#     source = pd.concat([df0, df2, df1, df4], axis=0, ignore_index=True)

#     source["close"] = source["val_"].cumsum()
#     source["open"] = source["close"].shift(1)
#     source.loc[len(source) - 1, "open"] = 0
#     source["open"].fillna(0, inplace=True)
#     return source.iloc[::-1]


def waterfall_chart(source, decimal=3):
    """Waterfall chart."""
    source = source.copy()
    for c in ["feature_value", "shap_value"]:
        source[c] = source[c].round(decimal).astype(str)
    source["feature_value"] = source["feature_value"].replace("nan", "")

    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            alt.X(
                "feature:O",
                sort=source["feature"].tolist(),
                axis=alt.Axis(labelLimit=120),
            ),
            alt.Y("open:Q", scale=alt.Scale(zero=False), title=""),
            alt.Y2("close:Q"),
            alt.Tooltip(["feature", "feature_value", "shap_value"]),
        )
    )
    color1 = bars.encode(
        color=alt.condition(
            "datum.open <= datum.close",
            alt.value("#FF0D57"),
            alt.value("#1E88E5"),
        ),
    )
    color2 = bars.encode(
        color=alt.condition(
            "datum.feature == 'Average Model Output' || "
            "datum.feature == 'Individual Observation'",
            alt.value("#F7E0B6"),
            alt.value(""),
        ),
    )
    text = bars.mark_text(
        align="center",
        baseline="middle",
        dy=-5,
        color="black",
    ).encode(
        text="feature_value:N",
    )
    return bars + color1 + color2 + text


def histogram_chart(source):
    """Histogram chart."""
    base = alt.Chart(source)
    chart = (
        base.mark_area(
            opacity=0.5,
            interpolate="step",
        )
        .encode(
            alt.X("Prediction:Q", bin=alt.Bin(maxbins=10), title="Prediction"),
            alt.Y("count()", stack=None),
        )
        .properties(
            width=280,
            height=200,
        )
    )
    return chart
