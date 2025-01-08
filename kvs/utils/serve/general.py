"""
General utility functions.
"""

from datetime import datetime
from os import getenv

import numpy as np
import pytz
from dateutil import parser


def parse_timestamp(ts_string):
    """
    Parse timestamp string to timestamp in SGT. Naive datetime are assumed to be SGT
    """
    return set_sgtz(parser.parse(ts_string))


def set_sgtz(timestamp):
    """Convert timestamp to SGT. Naive datetime are assumed to be SGT."""
    sgtz = pytz.timezone("Asia/Singapore")
    return sgtz.localize(timestamp)


def set_utcz(timestamp):
    """Convert timestamp to UTC. Naive datetime are assumed to be UTC."""
    utcz = pytz.timezone("UTC")
    return utcz.localize(timestamp)


def get_gain_pad(img1_shape, img0_shape):
    """Compute gain and pad for conversion from img0_shape to img1_shape."""
    # gain  = old / new
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])

    # wh padding
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    return gain, pad


def scale_coords(xys, gain, pad, inverse=True):
    """
    Rescale (x, y) from img0_shape to img1_shape,
    or from img1_shape to img0_shape if inverse = True.
    """
    xys = np.array(xys)
    if inverse:
        xys[:, 0] = ((xys[:, 0] - pad[0]) / gain).astype(int)
        xys[:, 1] = ((xys[:, 1] - pad[1]) / gain).astype(int)
        return xys

    xys[:, 0] = (xys[:, 0] * gain + pad[0]).astype(int)
    xys[:, 1] = (xys[:, 1] * gain + pad[1]).astype(int)
    return xys


def get_execution_time():
    """Get execution time using server time rounded down to the last hour if not provided."""
    execution_time = getenv("EXECUTION_TIME")
    if execution_time:
        return parse_timestamp(execution_time)

    # Naive server time assumed to be in UTC
    dt_now = datetime.now(pytz.timezone("Asia/Singapore"))
    # Round down to last hour
    return dt_now.replace(minute=0, second=0, microsecond=0)


def generate_filename(timestamp):
    return timestamp.strftime(f"%Y-%m-%d/%Y-%m-%dT%H_%M_%S")
