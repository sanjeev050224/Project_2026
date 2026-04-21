# 🦷 Tooth Decay Classification Using Dental X-Ray Images


---

##  Overview

**Tooth Decay Classification Using Dental X-Ray Images** is a deep learning-based final year project that automates the detection and classification of dental caries (tooth decay) from periapical X-ray images. The system leverages a fine-tuned **DenseNet-121** architecture enhanced with the **Convolutional Block Attention Module (CBAM)** to classify X-ray images into multiple severity categories — providing a faster, more consistent, and clinician-supportive diagnostic aid.

---

##  Problem Statement

Dental caries (tooth decay) is one of the most prevalent chronic diseases globally, yet early and accurate detection remains a challenge due to:

- High subjectivity in visual inspection by dentists
- Varying levels of expertise across practitioners
- Time-consuming manual analysis of X-ray images
- Limited access to specialist consultation in rural areas

This project addresses these challenges by building an intelligent classification system capable of detecting the **stage/severity of tooth decay** directly from dental X-ray images.

---

##  Model Architecture

### DenseNet-121 with CBAM (Convolutional Block Attention Module)

The backbone of this project is **DenseNet-121**, a densely connected convolutional network that reuses feature maps across layers, making it highly efficient for medical imaging tasks.

To further enhance feature extraction, we integrated **CBAM** — a lightweight attention mechanism that refines the model's focus on the most informative spatial and channel features in X-ray images.

```
Input X-Ray Image
       ↓
DenseNet-121 Backbone (Pre-trained on ImageNet)
       ↓
CBAM (Channel Attention + Spatial Attention)
       ↓
Global Average Pooling
       ↓
Fully Connected Layers + Dropout
       ↓
Softmax Output → Multi-class Classification
```

### Why DenseNet-121 with CBAM?
| Feature | Benefit |
|---|---|
| Dense connections | Better gradient flow, reduced vanishing gradient |
| Feature reuse | Fewer parameters, stronger representation |
| CBAM Channel Attention | Focuses on relevant feature channels |
| CBAM Spatial Attention | Highlights decay-affected regions in X-rays |
| Pre-trained weights | Faster convergence with limited medical data |

---

##  Classification Categories

The model classifies dental X-ray images into the following severity levels:

| Class | Label | Description |
|---|---|---|
| 0 | **No Decay** | Healthy tooth with no visible caries |
| 1 | **Mild & Moderate** | Caries reaching ename & dentin layer |
| 2 | **Severe** | Advanced decay, pulp involvement risk |


---

##  Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **PyTorch** | Model building, training & inference |
| **TensorFlow / Keras** | Data augmentation & preprocessing pipelines |
| **OpenCV** | X-ray image preprocessing (denoising, contrast enhancement) |
| **Google Colab** | Cloud-based GPU training environment |
| **Jupyter Notebook** | Experimentation & visualization |
| **Matplotlib / Seaborn** | Graphs, confusion matrix, training curves |
| **scikit-learn** | Classification report, metrics computation |

---

## Data Preprocessing Pipeline

X-ray images undergo a multi-step preprocessing pipeline before being fed to the model:

1. **Grayscale Conversion** — Dental X-rays are processed in grayscale
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — Enhances local contrast using OpenCV
3. **Noise Reduction** — Gaussian blur to remove sensor noise
4. **Resizing** — All images resized to 224 × 224 pixel with padding (DenseNet input size)
5. **Normalization** — Pixel values normalized to ImageNet mean and std
6. **Data Augmentation** — Random horizontal flips, rotations, brightness/contrast jitter to handle class imbalance

---

```
torch>=1.12.0
torchvision>=0.13.0
tensorflow>=2.9.0
keras>=2.9.0
opencv-python>=4.6.0
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
tqdm
```


##  Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| ResNet-50 | ~82% | Moderate | Moderate | Moderate |
| VGG-16 | ~79% | Low | Low | Low |
| DenseNet-121 | ~88% | Good | Good | Good |
| **DenseNet-121 + CBAM** | **Best** | **Best** | **Best** | **Best** |

>  **DenseNet-121 with CBAM achieved the best inference results among all compared models**, demonstrating superior capability in distinguishing subtle decay patterns in dental X-rays.



---

## Key Findings

- CBAM integration significantly improved the model's ability to localize decay regions in X-rays compared to vanilla DenseNet-121.
- Data augmentation was critical in handling the imbalanced distribution across severity classes.
- CLAHE-based preprocessing notably improved model convergence speed and classification stability.
- The model generalizes well across different X-ray acquisition settings and equipment types.

---

## 🔮 Future Scope

- [ ] Deploy as a **web application** using Flask/FastAPI for real-time clinical use
- [ ] Integrate **object detection** (e.g., YOLOv8) to localize decay bounding boxes on X-rays
- [ ] Expand dataset with more diverse sources for better generalization
- [ ] Explore **Transformer-based** architectures (e.g., Vision Transformer, Swin Transformer)
- [ ] Build a **mobile-friendly** diagnostic tool for remote/rural clinics

---

##  Authors

> Final Year M.sc Data Science

- **[Sreeram L V N Sanjeev]** — Model development, training & evaluation
- **[G Sasi Chandrika]** — Data collection,preprocessing & documentation

**Institution:** Gitam Deemed To Be University
**Department:** Mathematics
**Academic Year:** 2024–2026

---

##  Acknowledgements

- Dental dataset sourced from [National X-Ray Institute
- Annotations was done by Dr.Ramya Sri
- Project Guide - Dr.Abhijit Patil
- DenseNet architecture: [Huang et al., 2017 — Densely Connected Convolutional Networks]
- CBAM: [Woo et al., 2018 — Convolutional Block Attention Module]
- Pre-trained weights courtesy of **ImageNet** via PyTorch's `torchvision.models`

---

## 📄 License

This project is licensed under the **MIT License** — see the file for details.

---
