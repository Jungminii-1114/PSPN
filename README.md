# PSPNet: Pyramid Scene Parsing Network Implementation

This repository contains a PyTorch implementation of **PSPNet (Pyramid Scene Parsing Network)** based on the CVPR 2017 paper, [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105).

## 📌 Introduction
Semantic Segmentation involves classifying every pixel in an image into a specific category. While Fully Convolutional Networks (FCNs) have revolutionized this field, they often struggle to capture the **global context** of a scene due to their structural limitations.

This project aims to implement PSPNet to overcome these limitations, focusing on how the **Pyramid Pooling Module (PPM)** effectively aggregates global context information to improve segmentation performance.

---

## 💡 Theoretical Background

### 1. The Limitation of FCN: "Seeing the Trees but Missing the Forest"
Traditional FCN-based models excel at extracting local features (texture, shape) but suffer from a **limited receptive field**. Even with deep layers, the model may fail to understand the overall scene context.

This limitation often leads to the **"Mismatched Relationship"** problem:
> *Example:* If a boat is floating on a river, an FCN might misclassify the boat as a "car" based solely on its metallic texture and shape, ignoring the surrounding "water" context which makes the existence of a car logically impossible.

### 2. The Solution: Pyramid Pooling Module (PPM)
PSPNet introduces the **Pyramid Pooling Module** to incorporate **global context information** into the local features.



**Mechanism:**
The PPM aggregates context at four different pyramid scales to capture both global and local information:
1.  **Multi-Scale Pooling:** The feature map is pooled into four different scales (e.g., `1x1`, `2x2`, `3x3`, `6x6`).
    * `1x1` bin: Captures the **global prior** (the overall scene context).
    * Larger bins: Capture sub-region context.
2.  **Dimension Reduction:** A `1x1` convolution is applied to each pooled map to reduce the channel depth (bottleneck).
3.  **Upsampling:** The low-dimension feature maps are upsampled back to the size of the original feature map via bilinear interpolation.
4.  **Concatenation:** These upsampled context features are concatenated with the original local feature map.

**Conclusion:** By fusing local features with global priors, PSPNet can reason that *"since the surrounding area is water, the object inside must be a boat, not a car,"* significantly boosting segmentation accuracy.

---

## 🏗️ Model Architecture

The implementation follows the architecture described in the original paper:

* **Backbone:** `ResNet-50` or `ResNet-101`
    * **Dilated Convolution (Atrous Conv):** Applied to the last two blocks (`layer3`, `layer4`) to maintain a larger spatial resolution (Output Stride = 8) without losing the receptive field.
* **Neck (PPM):** Fuses features from four different pool scales.
* **Head:** Final convolution layer for pixel-wise classification.
* **Loss Function:** Standard Cross-Entropy Loss (optionally combined with Auxiliary Loss for stable training).

---
