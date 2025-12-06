#helpers.py
import cv2
import numpy as np

def preprocess_image(img_path, img_size=128):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image {img_path}")

    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0
    img = img.reshape(img_size, img_size, 1)  # ready for model

    return img
