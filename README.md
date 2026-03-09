# OCR Pipeline for Historical Documents

**RenAIssance Project**

A complete end-to-end OCR pipeline for printed historical documents combining traditional OCR with LLM-based post-correction.

## 🎯 Project Overview

This project demonstrates a practical OCR solution for digitizing historical documents using:
- **OpenCV** for image preprocessing
- **Tesseract OCR** for text extraction
- **LLM-based correction** for improving accuracy
- **Quantitative evaluation** using CER/WER metrics

## 🏗️ Architecture

```
Input Document (PDF/Image)
    ↓
Image Preprocessing (OpenCV)
    ↓
Text Region Detection
    ↓
OCR (Tesseract)
    ↓
LLM Post-Correction
    ↓
Evaluation (CER/WER)
```

## 📁 Repository Structure

```
gsoc/
├── ocr_pipeline.ipynb          # Main Jupyter notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   ├── images/                 # Input images
│   ├── pdfs/                   # Input PDFs
│   └── ground_truth/           # Ground truth text files
├── output/                     # Processing results
└── .env.example               # Environment variables template
```

## 🚀 Installation

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR**
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

3. **Poppler** (for PDF conversion)
   - Windows: Download from [GitHub](https://github.com/oschwartz10612/poppler-windows/releases)
   - Linux: `sudo apt-get install poppler-utils`
   - macOS: `brew install poppler`

### Setup

```bash
# Clone repository
git clone <your-repo-url>
cd gsoc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook ocr_pipeline.ipynb
```

## 📊 Usage

### Basic Pipeline

```python
from ocr_pipeline import ocr_pipeline, simulate_llm_correction, calculate_metrics

# Process single image
ocr_text, original, preprocessed = ocr_pipeline('data/images/document.png')

# Apply LLM correction
corrected_text = simulate_llm_correction(ocr_text)

# Evaluate
metrics = calculate_metrics(corrected_text, ground_truth)
```

### Batch Processing

```python
results = process_batch('data/images', 'data/ground_truth', use_llm=True)
```

## 🔧 Configuration

### Tesseract Configuration

Edit in notebook if needed:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### LLM Integration (Optional)

For real LLM correction using OpenAI:
1. Create `.env` file with `OPENAI_API_KEY=your_key`
2. Use `llm_correction_openai()` function

## 📈 Evaluation Metrics

- **Character Error Rate (CER)**: Character-level accuracy
- **Word Error Rate (WER)**: Word-level accuracy
- **Accuracy**: 1 - Error Rate

Lower CER/WER indicates better performance.

## 🧪 Testing with Sample Data

The notebook includes a function to create sample documents:

```python
sample_path, ground_truth = create_sample_document()
```

For real historical documents, place:
- Images in `data/images/`
- Ground truth text in `data/ground_truth/` (same filename, .txt extension)

## 🎓 Research Context

This pipeline is designed for the **RenAIssance project** evaluation, demonstrating:

1. **Traditional OCR**: Tesseract with optimized preprocessing
2. **Deep Learning Integration**: Framework for CNN-RNN models
3. **LLM Enhancement**: Post-processing for error correction
4. **Rigorous Evaluation**: Standard metrics for reproducibility

## 🔮 Future Enhancements

- [ ] Fine-tune Tesseract for historical fonts
- [ ] Implement CNN-RNN architecture (e.g., CRNN)
- [ ] Integrate transformer models (TrOCR, Donut)
- [ ] Advanced layout analysis
- [ ] Multi-language support
- [ ] Web interface for easy access

## 📚 References

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [Character Error Rate](https://en.wikipedia.org/wiki/Word_error_rate)


