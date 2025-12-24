# Handwritten Text Recognition (HTR) for Medical Prescriptions 💊

Built a robust deep learning solution to recognize and digitize handwritten text from medical prescriptions using a hybrid approach (CNN, LSTM, AlexNet, and OCR-BERT).

## Key Contributions & Engineering Decisions

* **Performance & Optimization:** Conducted a **comprehensive model audit** across three architectures, tracking standard metrics alongside **Inference Latency** to select a deployment candidate.
* **Deployment Engineering:** Optimized the final CNN model using **Post-Training Quantization (TFLite)** to ensure edge-device readiness. This resulted in:
    * **70% reduction in model size**
    * **3.3x average inference speedup**
* **Advanced Modeling:** Integrated initial **OCR (Pytesseract)** output with a **BERT Transformer model** for Masked Language Modeling (MLM) to refine and correct recognition errors in complex handwritten pharmaceutical terms.

## Files

* `doctor-s-prescription.ipynb`: The complete project notebook detailing data preprocessing, model training (CNN, LSTM, AlexNet), TFLite quantization, and the final performance audit.
* `README.md`: Project documentation explaining the objective, methodology, tech stack, usage instructions, and results.
* `.gitignore`: Specifies files and directories to be excluded from version control (e.g., checkpoints, virtual environments, cached files).
