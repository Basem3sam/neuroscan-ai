# 🧠 NeuroScan-AI

**Multi-pipeline brain tumor detection system using deep learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Intelligent medical imaging app that detects brain tumors from MRI/CT scans with **85-92% accuracy**. Features 5 classification pipelines (CNN, KNN, K-Means) and a professional GUI.

---

## ✨ Features

- 🏥 Medical-grade preprocessing with histogram equalization
- 🤖 5 classification pipelines: CNN, KNN, K-Means, and hybrids
- 🖼️ User-friendly GUI with real-time predictions
- 📊 85-92% accuracy on combined CT/MRI datasets
- 🔬 Modular architecture for research

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/basem3sam/neuroscan-ai.git
cd neuroscan-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Download datasets and organize as:

```
Dataset/
├── Brain Tumor CT scan Images/
│   ├── Healthy/
│   └── Tumor/
└── Brain Tumor MRI images/
    ├── Healthy/
    └── Tumor/
```

**Datasets**: [CT & MRI Dataset](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri)

Update path in `config.py`:

```python
DATA_ROOT = r"C:\path\to\your\Dataset"
```

### Training

```bash
python train.py
```

Expected output:

```
Epoch [10/10] Loss: 0.0542
✅ CNN Model saved
KNN Accuracy: 0.8654
K-Means Accuracy: 0.7823
```

### Run GUI

```bash
python gui.py
```

---

## 🔬 Pipelines

| Pipeline        | Description                  | Accuracy   | Best For         |
| --------------- | ---------------------------- | ---------- | ---------------- |
| **CNN** ⭐      | End-to-end deep learning     | **85-92%** | Production       |
| **KNN**         | Feature-based classification | 80-88%     | Interpretability |
| **K-Means**     | Unsupervised clustering      | 70-80%     | Exploration      |
| **CNN→KNN**     | CNN features + KNN           | 80-88%     | Research         |
| **K-Means→KNN** | Cluster distances + KNN      | 75-85%     | Experiments      |

---

## 📁 Project Structure

```
neuroscan-ai/
├── config.py              # Configuration
├── dataset.py             # Data loading
├── cnn.py                 # CNN model
├── knn.py                 # KNN classifier
├── kmeans_classifier.py   # K-Means
├── helpers.py             # Preprocessing
├── model_loader.py        # Inference
├── train.py               # Training script
├── gui.py                 # GUI application
└── models/                # Saved models
```

---

## 🛠️ Configuration

Edit `config.py`:

```python
IMAGE_SIZE = (128, 128)    # Input size
BATCH_SIZE = 16            # Training batch
EPOCHS = 10                # Training epochs
KNN_NEIGHBORS = 5          # K for KNN
```

---

## 📊 Results

**Test Performance (600 samples):**

| Metric    | CNN   | KNN   | K-Means |
| --------- | ----- | ----- | ------- |
| Accuracy  | 89.2% | 86.5% | 78.2%   |
| Precision | 90.1% | 87.3% | 79.8%   |
| Recall    | 88.5% | 85.9% | 76.5%   |

---

## 🐛 Troubleshooting

**CUDA out of memory**

```python
# Reduce batch size in config.py
BATCH_SIZE = 8
```

**Models not found**

```bash
python train.py  # Train first
```

**GUI too large**

```python
# Edit gui.py line 25
root.geometry("1024x768")
```

---

## 🤝 Contributing

Contributions welcome! Open issues or submit PRs.

**Ideas:**

- [ ] Data augmentation
- [ ] Grad-CAM visualization
- [ ] Web interface
- [ ] Multi-class classification
- [ ] DICOM support

---

## 📝 Citation

```bibtex
@software{neuroscan_ai_2024,
  author = {Basem Eaam},
  title = {NeuroScan-AI: Brain Tumor Detection System},
  year = {2024},
  url = {https://github.com/basem3sam/neuroscan-ai}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

⚠️ **Medical Disclaimer**: This software is for **research/education only**. NOT for clinical diagnosis. Always consult healthcare professionals.

---

## 👤 Author

**Your Name**

- GitHub: [@Basem3sam](https://github.com/Basem3sam)
- Email: basem.esam.omar@gmail.com

---

<div align="center">

**⭐ Star this repo if it helped you!**

[Report Bug](https://github.com/yourusername/neuroscan-ai/issues) • [Request Feature](https://github.com/yourusername/neuroscan-ai/issues)

</div>
