# %% [code]
# %% [code]
# %% [code]
# %% [code]
"""OCR helpers extracted from the notebook.
- `preprocess_image`
- `extract_text_with_ocr`
- `XYocr` to batch-process a folder and return texts + labels
"""
import os
import cv2
from PIL import Image
import pytesseract
import pandas as pd


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def extract_text_with_ocr(image_path):
    preprocessed_image = preprocess_image(image_path)
    img = Image.fromarray(preprocessed_image)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text


def XYocr(folder, dataset_root):
    lower = folder.lower()
    csv_path = os.path.join(dataset_root, folder, f"{lower}_labels.csv")
    df = pd.read_csv(csv_path)
    Y = df['MEDICINE_NAME']
    image_folder = os.path.join(dataset_root, folder, f"{lower}_words")
    image_paths = [os.path.join(image_folder, filename) for filename in df['IMAGE']]

    X = []
    batch_size = 50
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_texts = [extract_text_with_ocr(path) for path in batch_paths]
        X.extend(batch_texts)
        print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")
    return X, Y
