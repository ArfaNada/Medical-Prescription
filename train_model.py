# %% [code]
# %% [code]
# %% [code]
"""Train a simple sklearn model from a CSV and save the trained artifact.

Usage examples:
  python scripts/train_model.py --data data/clean.csv --output models/model.joblib

The script autodetects classification vs regression by looking at the target column's unique values.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from scripts.data_load import load_csv
from scripts.utils import set_seed, save_model


def train(data_path, output_model, target_column=None, test_size=0.2, random_state=42, n_estimators=100):
    set_seed(random_state)
    df = load_csv(data_path)
    if target_column is None:
        target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Decide classifier vs regressor
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        is_classification = False
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        is_classification = True

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    if is_classification:
        try:
            score = accuracy_score(y_val, preds)
            print(f"Validation accuracy: {score:.4f}")
        except Exception:
            print("Validation: could not compute accuracy.")
    else:
        try:
            mse = mean_squared_error(y_val, preds)
            print(f"Validation MSE: {mse:.4f}")
        except Exception:
            print("Validation: could not compute MSE.")

    save_model(model, output_model)
    print(f"Saved model to {output_model}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV input data")
    parser.add_argument("--output", required=True, help="Path to save trained model (joblib)")
    parser.add_argument("--target", default=None, help="Target column name (default: last column)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    train(args.data, args.output, target_column=args.target, test_size=args.test_size,
          random_state=args.random_state, n_estimators=args.n_estimators)


if __name__ == "__main__":
    cli()
