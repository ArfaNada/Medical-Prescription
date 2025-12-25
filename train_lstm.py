"""Train an LSTM model by streaming images from the processed data directory.

This script builds a tf.data pipeline that loads images lazily to keep memory low.
"""
import argparse
from pathlib import Path
import json
import pandas as pd
import tensorflow as tf

# Try to enable GPU memory growth to avoid allocation/copy initialization errors on shared GPUs
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--size', type=int, default=84)
    return p.parse_args()


def make_dataset(df, data_dir, size):
    filepaths = (data_dir / df['filepath']).astype(str).tolist() if 'filepath' in df.columns else (data_dir / 'train' / 'images' / df['IMAGE']).astype(str).tolist()
    labels = df['MEDICINE_NAME'].astype('int').tolist()
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        # Ensure the tensor has a known rank/channel count so resize can infer shape
        try:
            img.set_shape([None, None, 3])
        except Exception:
            pass
        img = tf.image.resize(img, [size, size])
        img = tf.cast(img, tf.float32) / 255.0
        # reshape to sequence: (sequence_length, feature_size)
        seq = tf.reshape(img, [size, size * 3])
        return seq, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def create_model(sequence_length, feature_size, output_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(sequence_length, feature_size), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_features, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')

    # ensure filepath column
    train_df['filepath'] = 'train/images/' + train_df['IMAGE'].astype(str)
    val_df['filepath'] = 'val/images/' + val_df['IMAGE'].astype(str)

    # load mapping to get num classes if available
    mapping_path = data_dir / 'mapping.json'
    if mapping_path.exists():
        import json
        with open(mapping_path) as f:
            mapping = json.load(f)
        num_classes = len(mapping)
    else:
        num_classes = len(train_df['MEDICINE_NAME'].unique())

    seq_len = args.size
    feature_size = args.size * 3

    train_ds = make_dataset(train_df, data_dir, args.size).shuffle(1024).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = make_dataset(val_df, data_dir, args.size).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = create_model(seq_len, feature_size, num_classes)
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output, save_format='h5')
    print('Saved LSTM model to', args.output)


if __name__ == '__main__':
    main()
