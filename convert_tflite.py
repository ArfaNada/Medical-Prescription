# %% [code]
# %% [code]
# %% [code]
"""Convert a saved Keras model (.h5) to a TFLite quantized model."""
import argparse
import tensorflow as tf
from pathlib import Path


def main(args):
    model_path = args.model
    out_path = args.output

    keras_model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    print('Saved TFLite to', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    main(args)
