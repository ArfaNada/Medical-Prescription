# Handwritten Text Recognition (HTR) for Medical Prescriptions 💊

Built a robust deep learning solution to recognize and digitize handwritten text from medical prescriptions using a hybrid approach (CNN, LSTM, AlexNet, and OCR-BERT).

## Key Contributions & Engineering Decisions

* **Performance & Optimization:** Conducted a **comprehensive model audit** across three architectures, tracking standard metrics alongside **Inference Latency** to select a deployment candidate.
* **Deployment Engineering:** Optimized the final CNN model using **Post-Training Quantization (TFLite)** to ensure edge-device readiness. This resulted in:

  - **70% reduction in model size**
  - **3.3x average inference speedup**

* **Advanced Modeling:** Integrated initial **OCR (Pytesseract)** output with a **BERT Transformer model** for Masked Language Modeling (MLM) to refine and correct recognition errors in complex handwritten pharmaceutical terms.

## Files

* [doctor-s-prescription.ipynb](doctor-s-prescription.ipynb) : Project notebook (exploration, preprocessing, experiments).
* [train_model.py](train_model.py) : Generic training entry point.
* [train_cnn.py](train_cnn.py) : CNN training script.
* [train_alexnet.py](train_alexnet.py) : AlexNet training script.
* [train_lstm.py](train_lstm.py) : LSTM training script.
* [run_kaggle.py](run_kaggle.py) : Helpers for Kaggle dataset runs (if applicable).
* [data_load.py](data_load.py) : Data loading and preprocessing utilities.
* [load_images.py](load_images.py) : Simple inference runner for image folders.
* [ocr_helper.py](ocr_helper.py) : OCR helper utilities (pytesseract integration, text cleanup).
* [convert_tflite.py](convert_tflite.py) : Convert and quantize trained model to TFLite.
* [utils.py](utils.py) : Misc utility functions used across scripts.
