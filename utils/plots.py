
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

sns.set_theme()


def plot_imgrid(ims, nrows_cols, titles=None, figsize=(12, 12)):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(
        fig,
        111,
        nrows_cols=nrows_cols,
        axes_pad=0.1,
        share_all=True,
    )
    grid[0].set_xticks([])
    grid[0].set_yticks([])

    for ax, im in zip(grid, ims):
        ax.imshow(im)

    if titles is not None:
        assert isinstance(titles, list)
        assert len(titles) == len(ims)
        for ax, title in zip(grid, titles):
            ax.set_title(title)
    return fig


def plot_history(train_history, test_history, ax=None, title=None, ylabel=None):
    """Plot Keras model history."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(train_history)
    ax.plot(test_history)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch")
    ax.legend(["train", "test"], loc="upper left")
    return ax


def plot_roc_curve(actual, pred, ax=None):
    """Plot ROC."""
    from sklearn import metrics

    fpr, tpr, _ = metrics.roc_curve(actual, pred)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC AUC = {:.4f}".format(metrics.roc_auc_score(actual, pred)))

    if ax is None:
        return fig
    return ax


def plot_pr_curve(actual, pred, ax=None):
    """Plot PR curve."""
    from sklearn import metrics

    precision, recall, _ = metrics.precision_recall_curve(actual, pred)

    if ax is None:
        fig, ax = plt.subplots()

    ax.step(recall, precision, color="b", alpha=0.2, where="post")
    ax.fill_between(recall, precision, alpha=0.2, color="b", step="post")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Avg precision = {:.4f}".format(metrics.average_precision_score(actual, pred))
    )

    if ax is None:
        return fig
    return ax

