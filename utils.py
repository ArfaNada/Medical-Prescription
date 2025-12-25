# %% [code]
# %% [code]
# %% [code]
"""Shared small utilities: seeding, model save/load, simple logging."""
import random
import numpy as np
import joblib
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def save_model(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path):
    return joblib.load(path)
