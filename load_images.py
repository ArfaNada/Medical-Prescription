# %% [code]
"""Load images from dataset CSVs, preprocess (resize + normalize), and save to `out_dir` as JPEGs and labels CSVs.

Saves:
- {out_dir}/train/images/*.jpg and train_labels.csv
- {out_dir}/val/images/*.jpg and val_labels.csv
- {out_dir}/test/images/*.jpg and test_labels.csv
- {out_dir}/mapping.json (label -> int)

This approach keeps memory low by writing files to disk incrementally.
"""
import argparse
from pathlib import Path
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from PIL import Image
import json
import os


def process_split(df, images_dir, src_dir, target_size=(84,84)):
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fname, label in zip(df['IMAGE'], df['MEDICINE_NAME']):
        src_path = Path(src_dir) / fname
        if not src_path.exists():
            continue
        img = load_img(str(src_path), target_size=target_size)
        arr = np.array(img, dtype=np.uint8)
        out_path = images_dir / fname
        Image.fromarray(arr).save(out_path)
        rows.append({'IMAGE': fname, 'MEDICINE_NAME': label})
    return pd.DataFrame(rows)


def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Read CSVs
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    df_test = pd.read_csv(args.test_csv)

    # Build label mapping from training set
    unique_labels = list(df_train['MEDICINE_NAME'].unique())
    mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Map labels to ints
    df_train['MEDICINE_NAME'] = df_train['MEDICINE_NAME'].map(mapping)
    df_val['MEDICINE_NAME'] = df_val['MEDICINE_NAME'].map(mapping)
    df_test['MEDICINE_NAME'] = df_test['MEDICINE_NAME'].map(mapping)

    # Save mapping
    with open(out / 'mapping.json', 'w') as f:
        json.dump(mapping, f)

    # Process and save images incrementally
    train_df_out = process_split(df_train, out / 'train' / 'images', args.train_dir, target_size=(args.size, args.size))
    val_df_out = process_split(df_val, out / 'val' / 'images', args.val_dir, target_size=(args.size, args.size))
    test_df_out = process_split(df_test, out / 'test' / 'images', args.test_dir, target_size=(args.size, args.size))

    train_df_out.to_csv(out / 'train_labels.csv', index=False)
    val_df_out.to_csv(out / 'val_labels.csv', index=False)
    test_df_out.to_csv(out / 'test_labels.csv', index=False)

    print('Saved processed images and labels to', out)

    # Save a small test sample array for notebook latency benchmarks
    try:
        import numpy as _np
        sample_paths = sorted((out / 'test' / 'images').glob('*'))[:64]
        samples = []
        for p in sample_paths:
            img = Image.open(p).convert('RGB')
            img = img.resize((args.size, args.size))
            arr = _np.asarray(img, dtype=_np.float32) / 255.0
            samples.append(arr)
        if samples:
            samples = _np.stack(samples, axis=0)
            # Save as NPZ under out_dir for notebook to pick up
            _np.savez_compressed(out / 'test_samples.npz', test=samples)
    except Exception:
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv', required=True)
    p.add_argument('--train-dir', required=True)
    p.add_argument('--val-csv', required=True)
    p.add_argument('--val-dir', required=True)
    p.add_argument('--test-csv', required=True)
    p.add_argument('--test-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--size', type=int, default=84)
    args = p.parse_args()
    main(args)
