import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pdpbox import pdp, info_plots

sns.set_theme()


def shap_summary_plot(
    shap_values,
    features,
    feature_names=None,
    max_display=None,
    plot_size=(12, 6),
    show=False,
):
    """Plot SHAP summary plot."""
    # TODO: convert to altair chart
    fig = plt.figure()
    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        max_display=max_display,
        plot_size=plot_size,
        show=show,
    )
    plt.tight_layout()
    return fig


def shap_dependence_plot(
    ind,
    shap_values=None,
    features=None,
    feature_names=None,
    interaction_index="auto",
    show=False,
    plot_size=(12, 6),
):
    """Plot dependence interaction chart."""
    # TODO: convert to altair chart
    fig, ax = plt.subplots(figsize=plot_size)
    shap.dependence_plot(
        ind,
        shap_values=shap_values,
        features=features,
        feature_names=feature_names,
        interaction_index=interaction_index,
        ax=ax,
        show=show,
    )
    plt.tight_layout()
    return fig


def pdp_plot(
    model,
    dataset,
    model_features,
    feature,
    feature_name,
    num_grid_points=10,
    xticklabels=None,
    plot_lines=False,
    frac_to_plot=1,
    plot_pts_dist=False,
    x_quantile=False,
    show_percentile=False,
):
    """Wrapper for pdp.pdp_plot. Uses pdp.pdp_isolate."""
    pdp_iso = pdp.pdp_isolate(
        model=model,
        dataset=dataset,
        model_features=model_features,
        feature=feature,
        num_grid_points=num_grid_points,
    )

    fig, axes = pdp.pdp_plot(
        pdp_iso,
        feature_name,
        plot_lines=plot_lines,
        frac_to_plot=frac_to_plot,
        plot_pts_dist=plot_pts_dist,
        x_quantile=x_quantile,
        show_percentile=show_percentile,
    )

    if xticklabels is not None:
        if plot_lines:
            _ = axes["pdp_ax"]["_count_ax"].set_xticklabels(xticklabels)
        else:
            _ = axes["pdp_ax"].set_xticklabels(xticklabels)
    return fig


def actual_plot(
    model,
    X,
    feature,
    feature_name,
    num_grid_points=10,
    xticklabels=None,
    show_percentile=False,
):
    """Wrapper for info_plots.actual_plot."""
    fig, axes, summary_df = info_plots.actual_plot(
        model=model,
        X=X,
        feature=feature,
        feature_name=feature_name,
        num_grid_points=num_grid_points,
        show_percentile=show_percentile,
        predict_kwds={},
    )

    if xticklabels is not None:
        _ = axes["bar_ax"].set_xticklabels(xticklabels)
    return fig, summary_df


def target_plot(
    df,
    feature,
    feature_name,
    target,
    num_grid_points=10,
    xticklabels=None,
    show_percentile=False,
):
    """Wrapper for info_plots.target_plot."""
    fig, axes, summary_df = info_plots.target_plot(
        df=df,
        feature=feature,
        feature_name=feature_name,
        target=target,
        num_grid_points=num_grid_points,
        show_percentile=show_percentile,
    )

    if xticklabels is not None:
        _ = axes["bar_ax"].set_xticklabels(xticklabels)
    return fig, summary_df


def pdp_interact_plot(
    model,
    dataset,
    model_features,
    feature1,
    feature2,
    plot_type="grid",
    x_quantile=True,
    plot_pdp=False,
):
    """Wrapper for pdp.pdp_interact_plot. Uses pdp.pdp_interact."""
    pdp_interact_out = pdp.pdp_interact(
        model=model,
        dataset=dataset,
        model_features=model_features,
        features=[feature1, feature2],
    )

    fig, _ = pdp.pdp_interact_plot(
        pdp_interact_out=pdp_interact_out,
        feature_names=[feature1, feature2],
        plot_type=plot_type,
        x_quantile=x_quantile,
        plot_pdp=plot_pdp,
    )
    return fig
