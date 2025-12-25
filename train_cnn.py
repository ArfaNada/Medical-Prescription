# %% [code]
"""Train the CNN model from processed image folder structure created by `load_images.py`.

Usage:
  python scripts/train_cnn.py --data-dir data/processed --output /kaggle/working/model.h5 --epochs 15 --batch-size 128
"""
import argparse
from pathlib import Path
import pandas as pd
import json
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


def create_model(input_shape=(84,84,3), num_classes=78):
    model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(16, (3,3), activation='relu',input_shape=input_shape),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dropout(0.3),
          tf.keras.layers.Dense(num_classes, activation='softmax')
      ])

    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=RMSprop(learning_rate=1e-4),
                    metrics=['accuracy'])
    return model


def main(args):
    data_dir = Path(args.data_dir)
    with open(data_dir / 'mapping.json') as f:
        mapping = json.load(f)
    num_classes = len(mapping)

    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')

    # ImageDataGenerator expects filenames relative to a directory
    train_df['filepath'] = 'train/images/' + train_df['IMAGE'].astype(str)
    val_df['filepath'] = 'val/images/' + val_df['IMAGE'].astype(str)

    # Ensure label column values are strings (ImageDataGenerator expects string labels)
    if train_df['MEDICINE_NAME'].dtype != object:
        train_df['MEDICINE_NAME'] = train_df['MEDICINE_NAME'].astype(str)
    if val_df['MEDICINE_NAME'].dtype != object:
        val_df['MEDICINE_NAME'] = val_df['MEDICINE_NAME'].astype(str)

    idg = ImageDataGenerator(rescale=1./255)
    train_gen = idg.flow_from_dataframe(train_df, directory=str(data_dir), x_col='filepath', y_col='MEDICINE_NAME',
                                        target_size=(args.size, args.size), color_mode='rgb', class_mode='sparse', batch_size=args.batch_size)
    val_gen = idg.flow_from_dataframe(val_df, directory=str(data_dir), x_col='filepath', y_col='MEDICINE_NAME',
                                      target_size=(args.size, args.size), color_mode='rgb', class_mode='sparse', batch_size=args.batch_size)

    model = create_model(input_shape=(args.size, args.size, 3), num_classes=num_classes)
    model.summary()

    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out), save_format='h5')
    # Save training history to a JSON next to the model for later inspection/plotting
    try:
        hist_path = out.with_suffix(out.suffix + '.history.json')
        import json as _json
        with open(hist_path, 'w') as _f:
            _json.dump(history.history, _f)
        print('Saved training history to', hist_path)
    except Exception as e:
        print('Failed to save training history:', e)

    print('Saved model to', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--size', type=int, default=84)
    args = p.parse_args()
    main(args)
