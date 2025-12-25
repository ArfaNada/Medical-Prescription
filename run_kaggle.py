"""Kaggle-friendly orchestrator to run preprocessing, training, and conversion steps.

Features:
- Detects default Kaggle dataset path; accepts custom `--dataset-root`.
- Skips steps if output artifacts already exist in `--work-dir`.
- Limits TF thread usage to reduce OOM risk on kernels.
- Runs each heavy step as a subprocess to keep the notebook kernel lightweight.
"""
import argparse
import os
import subprocess
from pathlib import Path
import sys


def find_script(script_name):
    """Locate a helper script on disk. Try (in order):
    - same directory as this orchestrator (useful when packaged under /kaggle/usr/lib/...)
    - repo-level `scripts/` folder
    - provided relative path (current working dir)
    - absolute matches under /kaggle/input/**
    Raises FileNotFoundError if none found.
    """
    candidates = []
    base = Path(__file__).parent
    # 1) same dir as this file
    candidates.append(base / script_name)
    # 2) parent repo scripts/ folder (common layout)
    candidates.append((base.parent / 'scripts' / script_name))
    # 3) relative path (current working dir)
    candidates.append(Path('scripts') / script_name)
    candidates.append(Path(script_name))
    # 4) scan /kaggle/usr/lib for matches (if running on Kaggle)
    try:
        import glob
        # Also check Kaggle utility install location (/kaggle/usr/lib/**)
        for m in glob.glob(f'/kaggle/usr/lib/**/{script_name}', recursive=True):
            candidates.append(Path(m))
    except Exception:
        pass

    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    raise FileNotFoundError(f"Could not locate helper script '{script_name}' (tried: {candidates})")


def configure_tf_threads():
    # Reduce threads to be kinder to shared Kaggle kernels
    os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '2')
    os.environ['MKL_NUM_THREADS'] = os.environ.get('MKL_NUM_THREADS', '2')
    # Ask TF to allow GPU memory growth to avoid OOM/init copy errors
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'true')


def run_cmd(cmd):
    print('>',' '.join(cmd))
    res = subprocess.run(cmd, shell=False)
    if res.returncode != 0:
        print('Command failed:', cmd)
        sys.exit(res.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', default="/kaggle/input/doctors-handwritten-prescription-bd-dataset/Doctor’s Handwritten Prescription BD dataset",
                   help='Root path for the dataset (Kaggle input path by default)')
    p.add_argument('--work-dir', default='/kaggle/working', help='Working dir to write processed data and models')
    p.add_argument('--out-processed', default='data/processed', help='Relative path to save processed images')
    p.add_argument('--skip-preprocess', action='store_true')
    p.add_argument('--skip-cnn', action='store_true')
    p.add_argument('--skip-lstm', action='store_true')
    p.add_argument('--skip-alexnet', action='store_true')
    args = p.parse_args()

    configure_tf_threads()

    dataset_root = Path(args.dataset_root)
    work_dir = Path(args.work_dir)
    processed_dir = Path(args.out_processed)

    train_csv = dataset_root / 'Training' / 'training_labels.csv'
    val_csv = dataset_root / 'Validation' / 'validation_labels.csv'
    test_csv = dataset_root / 'Testing' / 'testing_labels.csv'

    train_dir = dataset_root / 'Training' / 'training_words'
    val_dir = dataset_root / 'Validation' / 'validation_words'
    test_dir = dataset_root / 'Testing' / 'testing_words'

    # 1) Preprocess images (incremental) -> processed_dir
    if not args.skip_preprocess:
        if (processed_dir / 'train' / 'images').exists() and (processed_dir / 'train_labels.csv').exists():
            print('Processed images already present in', processed_dir, '- skipping preprocess')
        else:
            cmd = [sys.executable, str(find_script('load_images.py')),
                   '--train-csv', str(train_csv),
                   '--train-dir', str(train_dir),
                   '--val-csv', str(val_csv),
                   '--val-dir', str(val_dir),
                   '--test-csv', str(test_csv),
                   '--test-dir', str(test_dir),
                   '--out-dir', str(processed_dir)]
            run_cmd(cmd)

    # 2) Train CNN
    cnn_model_path = work_dir / 'prescription_classification_model.h5'
    if not args.skip_cnn:
        if cnn_model_path.exists():
            print('CNN model already exists at', cnn_model_path, '- skipping CNN training')
        else:
            cmd = [sys.executable, str(find_script('train_cnn.py')), '--data-dir', str(processed_dir), '--output', str(cnn_model_path)]
            run_cmd(cmd)

            # Convert to tflite after successful training
            tflite_path = work_dir / 'prescription_classification_quantized.tflite'
            cmd = [sys.executable, str(find_script('convert_tflite.py')), '--model', str(cnn_model_path), '--output', str(tflite_path)]
            run_cmd(cmd)

    # 3) Train LSTM (optional)
    lstm_model_path = work_dir / 'prescription_lstm.h5'
    if not args.skip_lstm:
        if lstm_model_path.exists():
            print('LSTM model already exists at', lstm_model_path, '- skipping LSTM training')
        else:
            cmd = [sys.executable, str(find_script('train_lstm.py')), '--data-dir', str(processed_dir), '--output', str(lstm_model_path)]
            run_cmd(cmd)

    # 4) Train AlexNet (optional)
    alexnet_model_path = work_dir / 'prescription_classification_alexnet.h5'
    if not args.skip_alexnet:
        if alexnet_model_path.exists():
            print('AlexNet model already exists at', alexnet_model_path, '- skipping AlexNet training')
        else:
            cmd = [sys.executable, str(find_script('train_alexnet.py')), '--data-dir', str(processed_dir), '--output', str(alexnet_model_path)]
            run_cmd(cmd)

    print('All requested steps finished. Artifacts in', work_dir)

if __name__ == '__main__':
    main()
