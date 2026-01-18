# Document Type Detection using Vision Transformer (ViT)

**🔗 Live Demo:** Experience the model in action: [Document Type Detector on Hugging Face Spaces](https://huggingface.co/spaces/bharath-shanmugasundaram/document-type-detector)

<p align="center">
  <a href="#system-design">
    <img src="https://img.shields.io/badge/System%20Design-View-blue?style=for-the-badge" />
  </a>
  <a href="#problem-exploration">
    <img src="https://img.shields.io/badge/Problem%20exploration-Explore-green?style=for-the-badge" />
  </a>
  <a href="#problem-approach-and-implementation">
    <img src="https://img.shields.io/badge/Implementation-Details-red?style=for-the-badge" />
  </a>
  <a href="#model-performance-and-other-metrics">
    <img src="https://img.shields.io/badge/Performance-Metrics-purple?style=for-the-badge" />
  </a>
</p>

## 🏆 Project Overview
This project implements a **production-grade Document Type Detection System** that accurately classifies scanned document images into five predefined categories based solely on their **visual layout and structural patterns**. By leveraging the cutting-edge Swin Transformer architecture, the system achieves **90%+ accuracy** without relying on OCR or textual content, making it language-agnostic and computationally efficient.

---

## 📋 Table of Contents
1.  [Key Features](#key-features)
2.  [Document Types](#document-types)
3.  [Model Architecture: Swin Transformer](#model-architecture-swin-transformer)
4.  [System Design](#system-design)
5.  [Problem Exploration](#problem-exploration)
6.  [Problem Approach and Implementation](#problem-approach-and-implementation)
7.  [Model Performance and Other Metrics](#model-performance-and-other-metrics)
8.  [Dataset](#dataset)
9.  [Quick Start](#quick-start)
10. [Future Enhancements](#future-enhancements)

---

## ✨ Key Features
- **High Accuracy**: 90% overall accuracy across 5 document types.
- **Fast Inference**: Optimized for quick predictions.
- **No OCR Dependency**: Pure vision-based, layout-aware approach.
- **Production Ready**: Includes a live web interface and API capabilities.

---

## 📄 Document Types
The model classifies documents into the following five categories:
1.  **Invoice**
2.  **Resume**
3.  **Bank Statement**
4.  **Insurance / Policy Document**
5.  **Government Form**

---

## 🤖 Model Architecture: Swin Transformer
This project utilizes a **Swin Transformer** (`swin_base_patch4_window7_224`) as its core architectural backbone, fine-tuned for the document classification task.

**Core Concepts:**
- **Hierarchical Architecture**: Unlike standard Vision Transformers (ViTs) that maintain a constant resolution, Swin Transformer creates a hierarchical feature map by merging image patches in deeper layers. This is crucial for understanding document layouts at various scales, from local text blocks to global page structure.
- **Shifted Windows**: The model divides the image into non-overlapping windows for self-attention computation, which is highly efficient. In subsequent layers, these windows are *shifted*, allowing cross-window connections and capturing broader contextual relationships essential for distinguishing document formats.
- **Linear Computational Complexity**: Self-attention is computed within local windows, leading to linear complexity with respect to image size. This makes Swin Transformer significantly more efficient for high-resolution document images compared to ViTs with global attention.

This architecture's ability to model both fine-grained details and long-range dependencies makes it exceptionally well-suited for document structure analysis.

---

<a id="system-design"></a>
## 🏗️ System Design

<img width="2166" height="1798" alt="image" src="https://github.com/user-attachments/assets/cb5828c6-7d12-49f5-8674-fa1c85943b66" />

The system follows a modular deep learning pipeline:
1.  **Input Processing**: Scanned document images are loaded and standardized.
2.  **Preprocessing**: Images are resized to 224x224, converted to tensors, and normalized.
3.  **Feature Extraction**: The pre-trained Swin Transformer backbone extracts hierarchical visual features.
4.  **Classification**: A custom classification head maps the extracted features to one of the five document type labels.
5.  **Output**: The system returns the predicted class along with confidence scores.

**Fine-Tuning Strategy**: A transfer learning approach was employed. The pre-trained Swin Transformer backbone was initially frozen, and only the final classification layer was trained. Subsequently, the last transformer block was unfrozen for further task-specific adaptation, allowing the model to refine its understanding of document-specific features without catastrophic forgetting of its general visual knowledge.

---

<a id="problem-exploration"></a>
## 🔍 Problem Exploration
The challenge was to automate the categorization of digitized documents based on visual structure. Traditional methods often rely on Optical Character Recognition (OCR), which introduces dependencies on text quality, language, and font, and adds significant processing overhead. A vision-based approach offers a more robust and generalizable solution, as the structural layout—tables, sections, logos, spacing—is a strong, language-agnostic indicator of document type. The primary task was to curate a suitable dataset and identify a model architecture capable of capturing these spatial and structural nuances effectively.

---

<a id="problem-approach-and-implementation"></a>
## ⚙️ Problem Approach and Implementation

### **Implementation Steps**
1.  **Dataset Curation**: Assembled and standardized a dataset of ~50,000 document images across 5 classes, split into training (40k) and validation (10k) sets.
2.  **Model Selection**: Chose the Swin Transformer for its efficiency and hierarchical modeling capability.
3.  **Training Configuration**:
    - **Optimizer**: AdamW with differential learning rates (1e-4 for the unfrozen backbone layer, 1e-3 for the classifier head).
    - **Loss Function**: Cross-Entropy Loss.
    - **Epochs**: 6.
    - **Batch Size**: 64.
4.  **Evaluation**: Set up a robust validation pipeline to track accuracy, loss, and generate detailed performance metrics.

### **Code Overview (Key Snippets)**
```
# Model Initialization
model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=5)

# Differential Fine-tuning
for param in model.parameters():
    param.requires_grad = False # Freeze backbone
for param in model.head.parameters():
    param.requires_grad = True  # Unfreeze classifier
for param in model.layers[-1].parameters():
    param.requires_grad = True  # Unfreeze last stage

# Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {"params": model.layers[-1].parameters(), "lr": 1e-4},
    {"params": model.head.parameters(), "lr": 1e-3},
])
```

---

<a id="model-performance-and-other-metrics"></a>
## 📊 Model Performance and Other Metrics

### **Detailed Classification Report**
The model was evaluated on a held-out test set of 9,984 samples. The results are summarized below:

| Document Type | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bank Statement** | 0.88 | 0.90 | 0.89 | 1,996 |
| **Government Forms** | 0.84 | 0.84 | 0.84 | 1,997 |
| **Insurance** | 0.87 | 0.84 | 0.85 | 1,998 |
| **Invoice** | 0.94 | 0.95 | 0.94 | 1,996 |
| **Resume** | 0.98 | 0.99 | 0.98 | 1,997 |
| | | | | |
| **Accuracy** | | | **0.90** | 9,984 |
| **Macro Avg** | 0.90 | 0.90 | 0.90 | 9,984 |
| **Weighted Avg** | 0.90 | 0.90 | 0.90 | 9,984 |

### **Analysis**
- **Overall Performance**: The model achieved a strong **90% accuracy**, demonstrating its effectiveness.
- **Class-wise Insight**: The **Resume** and **Invoice** classes show near-perfect performance (F1 > 0.94), likely due to their highly distinctive and consistent visual layouts. Slightly lower scores for **Government Forms** and **Insurance** documents suggest these classes may have more intra-class visual variation or share structural similarities with other types.
- **Conclusion**: The Swin Transformer-based approach successfully solves the document classification problem with high accuracy, validating the hypothesis that visual structure is a powerful feature for this task.

---

## 📁 Dataset
- **Source**: Custom-curated public datasets.
- **Total Images**: Approximately 50,000.
- **Split**: 40,000 for training, 10,000 for validation/testing.
- **Availability**: The dataset is publicly available on Hugging Face: [`bharath-shanmugasundaram/Document-Type-Detection`](https://huggingface.co/datasets/bharath-shanmugasundaram/Document-Type-Detection).

---

<a id="future-enhancements"></a>
## 🚀 Future Enhancements
Potential next steps include:
- Expanding the number of document classes.
- Experimenting with other vision transformer architectures.
- Implementing ensemble methods for higher robustness.
- Deploying the model as a microservice API.

---
**Built with ❤️ using PyTorch, Swin Transformers, and Hugging Face.**
