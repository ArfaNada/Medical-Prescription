# %% [code]
# %% [code]
# %% [code]
"""Data loading and intermediate I/O helpers for notebook->script workflow."""
from pathlib import Path
import pandas as pd
import joblib


def load_csv(path, nrows=None):
    path = Path(path)
    return pd.read_csv(path, nrows=nrows)


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path):
    path = Path(path)
    return joblib.load(path)


def save_dataframe(df, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
