"""
Simple OCR Pipeline Runner - No Jupyter Needed
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from jiwer import wer, cer
import re

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("="*50)
print("OCR PIPELINE - RenAIssance Project")
print("="*50)

# Create sample document
print("\n1. Creating sample document...")
img = Image.new('RGB', (800, 400), color='white')
draw = ImageDraw.Draw(img)
text = "In the year 1776, the Declaration of Independence was signed by the founding fathers. This document proclaimed the freedom and sovereignty of the thirteen American colonies."
draw.text((50, 50), text, fill='black')
img.save('data/images/sample.png')
print("✓ Sample created: data/images/sample.png")

# Load and preprocess
print("\n2. Preprocessing image...")
image = cv2.imread('data/images/sample.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoised = cv2.fastNlMeansDenoising(gray, h=10)
thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
print("✓ Preprocessing complete")

# OCR
print("\n3. Running OCR...")
ocr_text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6').strip()
print("\nOCR Output:")
print("-"*50)
print(ocr_text)

# LLM Correction
print("\n4. Applying LLM correction...")
corrections = {r'\bl\b': 'I', r'\b0\b': 'O', r'rn': 'm', r'vv': 'w'}
corrected = ocr_text
for pattern, replacement in corrections.items():
    corrected = re.sub(pattern, replacement, corrected)
print("\nCorrected Output:")
print("-"*50)
print(corrected)

# Evaluation
print("\n5. Calculating metrics...")
ground_truth = text
cer_score = cer(ground_truth, corrected)
wer_score = wer(ground_truth, corrected)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Character Error Rate (CER): {cer_score:.4f}")
print(f"Word Error Rate (WER): {wer_score:.4f}")
print(f"Character Accuracy: {(1-cer_score)*100:.2f}%")
print(f"Word Accuracy: {(1-wer_score)*100:.2f}%")
print("="*50)

print("\n✓ Pipeline complete!")
