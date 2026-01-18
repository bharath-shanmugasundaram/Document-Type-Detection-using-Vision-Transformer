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

<a id="problem-exploration"></a>
## 🔍 Problem Exploration & Motivation

The objective of this project is to automate the categorization of digitized documents by leveraging their **visual structure and layout characteristics**, rather than relying on textual content. Through extensive exploration, it became evident that the structural composition of a document often provides stronger and more consistent signals for classification than the extracted text itself.

Traditional document classification systems are typically built around **Optical Character Recognition (OCR)** pipelines. While effective in controlled settings, these approaches introduce several practical limitations:
- High dependency on text quality, scan resolution, fonts, and language
- Sensitivity to noise, skew, and low-quality scans
- Additional preprocessing stages that increase system complexity and inference latency

During problem analysis, it was observed that most document types can be reliably distinguished using **layout-driven visual cues**, such as:
- Arrangement of text blocks and sections
- Presence and placement of tables, logos, and headers
- Spacing, alignment, and overall page structure

These patterns remain largely **language-agnostic and content-independent**, making a vision-based approach significantly more robust across real-world document variations.

A key challenge in adopting this approach was the need to:
- Curate a diverse and representative dataset that captures variations in document formats, layouts, and scan conditions
- Identify a model architecture capable of learning both **fine-grained local details** and **global spatial relationships** within a document

Transformer-based vision models emerged as a natural fit for this problem. In particular, Vision Transformers (ViTs) demonstrated strong potential due to their ability to model long-range dependencies and capture global context across an entire page. This insight ultimately guided the architectural choices made in this project, laying the foundation for a scalable, OCR-free document classification system.

---

<a id="problem-approach-and-implementation"></a>
## ⚙️ Problem Approach and Implementation

This project adopts a structured, experimentation-driven approach to build a robust document type classification system using visual cues alone. The implementation emphasizes scalability, efficiency, and generalization across diverse document layouts.

### **Approach Overview**
Rather than relying on OCR pipelines, the problem was reframed as a **pure vision-based classification task**, where document layout and structural patterns serve as the primary discriminative signals. A transformer-based architecture was selected to effectively capture both local and global spatial relationships within document images.

---

### **Implementation Details**

#### 1. Dataset Curation and Preparation
- A dataset of approximately **50,000 document images** spanning **five distinct document categories** was curated.
- The dataset includes variations in layout, formatting, and scan quality to improve real-world robustness.
- Data was split into:
  - **Training set**: ~40,000 images
  - **Validation set**: ~10,000 images
- All images were standardized through resizing and normalization to ensure consistency during training.

---

#### 2. Model Architecture Selection
- The **Swin Transformer** was selected as the backbone architecture due to its:
  - Hierarchical feature representation
  - Efficient window-based self-attention
  - Strong performance on structured visual data
- A custom classification head was added to adapt the pretrained model to the document classification task.

---

#### 3. Training Strategy and Configuration
- A **transfer learning** approach was employed to leverage pretrained visual knowledge.
- The backbone was initially frozen, and selective fine-tuning was applied to:
  - The classification head
  - The final transformer stage
- Training configuration:
  - **Optimizer**: AdamW, chosen for its effectiveness with transformer-based models
  - **Differential Learning Rates**:
    - `1e-4` for the unfrozen backbone layers
    - `1e-3` for the classification head
  - **Loss Function**: Cross-Entropy Loss for multi-class classification
  - **Batch Size**: 64
  - **Epochs**: 6
- This configuration balances convergence speed with stability while reducing the risk of overfitting.

---

#### 4. Evaluation and Validation
- A validation pipeline was integrated to monitor model performance after each epoch.
- Metrics tracked include:
  - Training and validation loss
  - Classification accuracy
- Post-training evaluation includes:
  - Confusion matrix analysis
  - Class-wise performance metrics
- These evaluations provide insights into both overall performance and category-specific behavior.

---

This approach ensures a clear separation between data preparation, model configuration, training logic, and evaluation, resulting in a reproducible and extensible implementation.

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

## 📄 Document Types
The model classifies documents into the following five categories:
1.  **Invoice**
2.  **Resume**
3.  **Bank Statement**
4.  **Insurance / Policy Document**
5.  **Government Form**

---

## 🤖 Model Architecture: [`Swin_Transformer`](https://github.com/microsoft/Swin-Transformer)
This project utilizes a **Swin Transformer** (`swin_base_patch4_window7_224`) as its core architectural backbone, fine-tuned for the document classification task.
<img width="2386" height="1312" alt="image" src="https://github.com/user-attachments/assets/37341143-4b04-4e7a-875e-d0dbd4f15151" />

### 🧠 Core Concepts Behind Swin Transformer

The Swin Transformer architecture introduces several key innovations that make it particularly effective for document structure understanding, where both fine-grained details and global layout cues are critical.

- **Hierarchical Feature Representation**  
  Unlike standard Vision Transformers (ViTs) that operate at a fixed spatial resolution, the Swin Transformer constructs a **hierarchical feature pyramid** by progressively merging image patches in deeper layers.  
  This design enables the model to:
  - Capture fine-level visual elements such as text blocks and logos in early stages
  - Learn high-level structural patterns like page layout and section organization in later stages  
  Such multi-scale representation is essential for robust document layout analysis.

- **Shifted Window-Based Self-Attention**  
  Self-attention is computed within local, non-overlapping windows to maintain efficiency. In alternating layers, these windows are **shifted**, allowing information to flow across window boundaries.  
  This mechanism:
  - Preserves computational efficiency
  - Enables cross-region interaction
  - Captures broader contextual relationships across the document  
  As a result, the model effectively distinguishes document formats that differ in layout rather than textual content.

- **Linear Computational Complexity**  
  By restricting self-attention to local windows, the Swin Transformer achieves **linear computational complexity with respect to image size**, in contrast to the quadratic complexity of global-attention ViTs.  
  This makes the architecture:
  - Scalable to high-resolution document images
  - Suitable for real-world deployment scenarios
  - Efficient without sacrificing representational power

Together, these architectural principles allow the Swin Transformer to simultaneously model **local visual details** and **long-range structural dependencies**, making it exceptionally well-suited for vision-based document type classification.

---

<a id="system-design"></a>
## 🏗️ System Design

<img width="2166" height="1798" alt="image" src="https://github.com/user-attachments/assets/cb5828c6-7d12-49f5-8674-fa1c85943b66" />

The system is designed as a modular deep learning pipeline for robust document type classification based purely on visual structure and layout information.

#### 1. Input Processing
- Scanned document images are ingested as input.
- Both single-page images and pages extracted from multi-page documents are supported.
- Inputs are standardized to ensure consistency across diverse document sources.

#### 2. Preprocessing
- Images are resized to a fixed resolution of **224 × 224** pixels.
- Data is converted into tensors and normalized using ImageNet statistics.
- This step ensures compatibility with pretrained transformer weights and stable training dynamics.

#### 3. Feature Extraction
- A **pre-trained Swin Transformer** is used as the backbone architecture.
- The hierarchical window-based attention mechanism enables:
  - Capture of both local layout patterns and global document structure
  - Robust understanding of tables, sections, headers, and spatial alignment
- The backbone serves as a powerful feature extractor for document-level representations.

#### 4. Classification Head
- A custom classification head is attached to the transformer backbone.
- Extracted features are mapped to one of the **five predefined document type classes**.
- A softmax layer produces class probabilities, enabling confidence-based predictions.

#### 5. Output
- The system returns:
  - The predicted document type
  - Associated confidence scores for each class
- This design allows easy integration into downstream validation or decision-making systems.

---

### 🔁 Fine-Tuning Strategy

A **transfer learning–based fine-tuning strategy** was employed to balance performance and training efficiency:

- Initially, the **Swin Transformer backbone was frozen**, and only the classification head was trained.
  - This preserves the general visual knowledge learned during large-scale pretraining.
- In the subsequent phase, the **final transformer block was selectively unfrozen**.
  - This allows the model to adapt to document-specific visual patterns such as layout, spacing, and structural cues.
- This staged unfreezing strategy improves task-specific performance while avoiding **catastrophic forgetting** and overfitting.

---

**Design Highlights**
- Vision-only, OCR-free document classification
- Modular and extensible architecture
- Efficient transfer learning with selective fine-tuning
- Layout-aware transformer-based feature extraction


---

## ✨ Key Features
- **High Accuracy**: 90% overall accuracy across 5 document types.
- **Fast Inference**: Optimized for quick predictions.
- **No OCR Dependency**: Pure vision-based, layout-aware approach.
- **Production Ready**: Includes a live web interface and API capabilities.

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

# 📊 Dataset Curation: From Raw Collection to Refined Repository

## 🎯 The Data Foundation Challenge
Building an accurate document classifier starts with quality data. Since no suitable dataset existed, I undertook the comprehensive task of **collecting, cleaning, and structuring** a robust dataset from scratch.

## 🔄 Data Pipeline: Collection to Deployment

### **Phase 1: Multi-Source Aggregation**
- **Initial Sources**: Began with public datasets (RVL-CDIP, DocBank)
- **Targeted Collection**: Added specialized samples through web scraping:
  - Government portal forms and templates
  - Business document repositories
  - Publicly shared professional templates
- **Quality Focus**: Prioritized **visual clarity and layout diversity** over sheer volume

### **Phase 2: Rigorous Cleaning & Validation**
- **Manual Review**: Personally inspected thousands of images to ensure:
  - Correct categorization (avoiding mislabeled documents)
  - Sufficient resolution for model training
  - Real-world applicability (not just perfect templates)
- **Consistency Checks**: Established clear guidelines for ambiguous cases
- **Balance Achievement**: Worked to create equal representation across all 5 classes

## 📈 Strategic Dataset Split
Implemented a **structured, stratified split** to ensure fair evaluation:
- **Training**: 40,000 images (80%)
- **Validation**: 5,000 images (10%) 
- **Test**: 5,000 images (10%)

**Key Insight**: This balanced split prevented class imbalance issues and provided reliable performance metrics.

## 🚀 Hugging Face Deployment
### **Preparation Steps:**
1. **Optimized Formatting**: Converted to efficient, standardized structure
2. **Comprehensive Documentation**: Created detailed dataset cards with usage examples
3. **Easy Integration**: Structured for seamless loading via Hugging Face's `datasets` library

### **Public Repository:**
The final curated dataset is available at:  
**[`bharath-shanmugasundaram/Document-Type-Detection`](https://huggingface.co/datasets/bharath-shanmugasundaram/Document-Type-Detection)**

## 💡 Core Learning Outcomes
- **ML Reality Check**: Experienced firsthand that **data preparation consumes majority of project time**
- **Quality Over Quantity**: Learned that 10,000 clean samples per class outperforms 50,000 noisy ones
- **Bias Awareness**: Developed skills to identify and mitigate dataset biases
- **Reproducibility**: Understood the importance of meticulous data documentation

This hands-on data curation was **fundamental to the project's success**—transforming theoretical knowledge into practical understanding of what makes machine learning systems work in reality.

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
