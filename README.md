# image-super-resolution

---

## 1. Introduction
Image super-resolution is a fundamental problem in computer vision where the goal is to reconstruct **high-resolution images** from **low-resolution inputs**. In this report, we explore different **neural network architectures** for super-resolution and evaluate their performance using **PSNR (Peak Signal-to-Noise Ratio)** and other relevant metrics.

---

## 2. Dataset Preparation
### Dataset: PascalVOC 2007
- **Image Sizes:**
  - **X (Input):** 72 × 72 × 3
  - **Y_mid (Intermediate Target):** 144 × 144 × 3
  - **Y_large (Final Target):** 288 × 288 × 3
- **Data Split:**
  - **Training:** 80%
  - **Validation:** 20%
- **Data Loader:**
  - Images were dynamically resized and normalized for model training.

---

## 3. Model Architectures and Training
We experimented with multiple **deep learning architectures** to improve image super-resolution, progressively refining our models.

### 3.1 Baseline Model: Fully Convolutional Network (CNN with Upsampling)
- **Architecture:**
  - **Conv2D (64 filters, 3×3, ReLU)**: Extracts basic edge and texture features.
  - **Conv2D (64 filters, 3×3, ReLU)**: Further refines features.
  - **Bilinear Upsampling (×2 factor)**: Doubles spatial resolution.
  - **Conv2D (3 filters, 1×1, linear activation)**: Produces final RGB image.
- **Output:** 144×144×3

### 3.2 Two-Stage Upsampling Model (Multi-Scale Output)
- **Architecture:**
  - **Upsampling to 144×144** → First prediction.
  - **Upsampling to 288×288** → Final high-resolution prediction.
- **Output:** Both 144×144 and 288×288 images.

### 3.3 Residual Learning Model (ResNet-Based)
- **Key Idea:** Learns the **residual difference** between low- and high-resolution images.
- **Architecture:**
  - **Residual Blocks (skip connections).**
  - **Two-stage upsampling (144×144, then 288×288).**
- **Output:** 144×144 and 288×288 images.

### 3.4 Dilated Convolution Model
- **Key Idea:** Uses **dilated convolutions** to increase receptive field **without increasing parameters**.
- **Architecture:**
  - **Dilated Conv Block (Dilation Rates: 1, 2, 4)**.
  - **Skip Connection before activation.**
  - **Upsampling to 144×144 → Upsampling to 288×288**.
- **Output:** 144×144 and 288×288 images.

### 3.5 Feature Extraction with Pretrained VGG16
- **Key Idea:** Leverages pretrained **VGG16** features to enhance reconstruction quality.
- **Architecture:**
  - **VGG16 Feature Extraction** (block1_conv2).
  - **Concatenation with convolutional outputs.**
  - **Upsampling to 144×144 → 288×288.**
- **Output:** 144×144 and 288×288 images.

### 3.6 Feature Extraction with CLIP
- **Key Idea:** Uses **CLIP embeddings**, which contain **semantic understanding of images**.
- **Architecture:**
  - **CLIP Feature Extraction.**
  - **Concatenation with convolutional outputs.**
  - **Upsampling to 144×144 → 288×288.**
- **Output:** 144×144 and 288×288 images.

---

## 4. CLIP Similarity Matrix and Image Analysis
- **Objective:** Measure **alignment between generated images and textual descriptions** using **CLIP cosine similarity**.
- **Generated Image Descriptions:**
  - Example: “Fishing boat sailing on a calm ocean.”
- **Evaluation:**
  - Computed a **similarity matrix** to compare text and image embeddings.
  - Higher similarity values indicate a better match.

---

## 5. Evaluation Metrics
| Model | PSNR (Mid) | PSNR (Large) | MSE (Mid) | MSE (Large) | Training Time |
|------------|------------|------------|------------|------------|------------|
| **Baseline CNN** | 74.72 | XX.XX | 0.0022 | XX.XX | 543.34 sec |
| **Two-Stage Upsampling** | 75.29 | 71.74 | 0.0019 | 0.0044 | 589.03 sec |
| **Residual Connections** | 75.60 | 72.40 | 0.00179 | 0.00382 | 668.69 sec |
| **Dilated Convolution** | 75.50 | 72.20 | 0.00187 | 0.00395 | 718.75 sec |
| **VGG16 Feature Extraction** | 75.70 | 72.20 | 0.00178 | 0.00397 | 624.60 sec |
| **CLIP Feature Extraction** | 75.10 | 71.80 | 0.00201 | 0.00433 | 832.01 sec |

---

## 6. Analysis of Results
### Best Performing Model (PSNR & MSE)
- **VGG16 Feature Extraction** achieved the highest PSNR (75.70 Mid, 72.20 Large) and lowest MSE (0.00178 Mid).
- **Residual Connections** followed closely, performing best on **large images**.

### Computational Cost vs. Performance Trade-off
- **Baseline CNN** was **fastest** (~543 sec), but had lower PSNR.
- **Dilated Convolution & VGG16** had **higher training time**, but good performance.
- **CLIP Feature Extraction** had **highest training time** (~832 sec), but did **not outperform VGG16-based models**.

### Effectiveness of Feature Extraction
- **VGG16 Feature Extraction performed best** → Deep visual features improved super-resolution.
- **CLIP Feature Extraction did not surpass traditional methods**, likely due to its generalization for semantic understanding rather than pixel-level super-resolution.

---

## 7. Final Conclusions
- **Best model:** **VGG16 Feature Extraction**.
- **Best balance of speed & quality:** **Residual Learning + Two-Stage Upsampling**.
- **Baseline CNN is the fastest**, but lacks quality.
- **CLIP Feature Extraction did not outperform traditional deep learning** for super-resolution.

---
