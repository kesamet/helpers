"""
Metric helpers. Adapted from scikit-plot
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder


def cumulative_gain_curve(y_true, y_score, pos_label=None):
    """This function generates the points necessary to plot the Cumulative Gain
    Note: This implementation is restricted to the binary classification task.
    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.
        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).
        pos_label (int or str, default=None): Label considered as positive and
            others are considered negative
    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.
        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.
    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains


def cumulative_lift_curve(y_true, y_score, pos_label=None):
    """This function generates the points necessary to plot the Cumulative Lift
    Note: This implementation is restricted to the binary classification task.
    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.
        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).
        pos_label (int or str, default=None): Label considered as positive and
            others are considered negative
    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.
        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.
    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    percentages, gains = cumulative_gain_curve(y_true, y_score, pos_label=pos_label)
    percentages = percentages[1:]
    gains = gains[1:] / percentages
    return percentages, gains


def binary_ks_curve(y_true, y_probas):
    """This function generates the points necessary to calculate the KS
    Statistic curve.
    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.
        y_probas (array-like, shape (n_samples)): Probability predictions of
            the positive class.
    Returns:
        thresholds (numpy.ndarray): An array containing the X-axis values for
            plotting the KS Statistic plot.
        pct1 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.
        pct2 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.
        ks_statistic (float): The KS Statistic, or the maximum vertical
            distance between the two curves.
        max_distance_at (float): The X-axis value at which the maximum vertical
            distance between the two curves is seen.
        classes (np.ndarray, shape (2)): An array containing the labels of the
            two classes making up `y_true`.
    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The KS Statistic
            is only relevant in binary classification.
    """
    y_true, y_probas = np.asarray(y_true), np.asarray(y_probas)
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
    if len(lb.classes_) != 2:
        raise ValueError(
            "Cannot calculate KS statistic for data with " "{} category/ies".format(
                len(lb.classes_)
            )
        )
    idx = encoded_labels == 0
    data1 = np.sort(y_probas[idx])
    data2 = np.sort(y_probas[np.logical_not(idx)])

    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []
    while ctr1 < len(data1) or ctr2 < len(data2):
        # Check if data1 has no more elements
        if ctr1 >= len(data1):
            current = data2[ctr2]
            while ctr2 < len(data2) and current == data2[ctr2]:
                ctr2 += 1

        # Check if data2 has no more elements
        elif ctr2 >= len(data2):
            current = data1[ctr1]
            while ctr1 < len(data1) and current == data1[ctr1]:
                ctr1 += 1

        else:
            if data1[ctr1] > data2[ctr2]:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1

            elif data1[ctr1] < data2[ctr2]:
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

            else:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

        thresholds.append(current)
        pct1.append(ctr1)
        pct2.append(ctr2)

    thresholds = np.asarray(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))

    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])
        pct1 = np.insert(pct1, 0, [0.0])
        pct2 = np.insert(pct2, 0, [0.0])
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])
        pct1 = np.append(pct1, [1.0])
        pct2 = np.append(pct2, [1.0])

    differences = pct1 - pct2
    ks_statistic = np.max(differences)
    max_distance_at = thresholds[np.argmax(differences)]
    return thresholds, pct1, pct2, ks_statistic, max_distance_at, lb.classes_
