"""
Utility functions for exporting and loading
"""
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Union

import pandas as pd


def make_dir(folder: Path) -> None:
    """Make directory, or replace if exists."""
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def export_csv(data: pd.DataFrame, output_path: Union[Path, str]) -> None:
    data.to_csv(output_path, index=False)


def load_csv(filepath: Union[Path, str]) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(e)


def export_json(data: dict, output_path: Union[Path, str]) -> None:
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Union[Path, str]) -> dict:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def export_pkl(python_object: Any, filepath: Union[Path, str]) -> None:
    if not python_object:
        raise TypeError(
            "python_object must be non-zero, non-empty, and not None"
        )
    with open(filepath, "wb") as f:
        pickle.dump(python_object, f)


def load_pkl(filepath: Union[Path, str]) -> Any:
    with open(filepath, "rb") as f:
        return pickle.load(f)
