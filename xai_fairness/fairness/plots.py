import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import compute_fairness_measures

sns.set_theme()


def fairness_summary(aif_metric, threshold=0.2):
    """Fairness charts wrapper function."""
    lower = 1 - threshold
    upper = 1 / lower
    print(
        "Model is considered fair for the metric when "
        f"**ratio is between {lower:.2f} and {upper:.2f}**."
    )

    fmeasures = compute_fairness_measures(aif_metric)
    fmeasures["Fair?"] = fmeasures["Ratio"].apply(
        lambda x: "Yes" if lower < x < upper else "No"
    )

    # display(fmeasures.iloc[:3].style.applymap(color_red, subset=["Fair?"]))

    fig_cm = plot_confusion_matrix_by_group(aif_metric)
    return fmeasures, fig_cm


def plot_confusion_matrix_by_group(aif_metric, figsize=(14, 4)):
    """Plot confusion matrix by group."""

    def _cast_cm(x):
        return np.array([[x["TN"], x["FP"]], [x["FN"], x["TP"]]])

    cmap = plt.get_cmap("Blues")
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for i, (privileged, title) in enumerate(
        zip([None, True, False], ["all", "privileged", "unprivileged"])
    ):
        cm = _cast_cm(
            aif_metric.binary_confusion_matrix(privileged=privileged)
        )
        sns.heatmap(cm, cmap=cmap, annot=True, fmt="g", ax=axs[i])
        axs[i].set_xlabel("predicted")
        axs[i].set_ylabel("actual")
        axs[i].set_title(title)
    return fig
