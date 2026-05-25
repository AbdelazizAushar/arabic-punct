# Arabic Punctuation Prediction

A comprehensive deep learning project for predicting and restoring punctuation marks in Arabic sentences using two complementary approaches.

## 📋 Overview

This project implements and compares two distinct approaches for Arabic punctuation prediction and restoration:

1. **Static Embedding Models + Deep Learning Sequence Labeling**
   - Word embeddings: AraVec (CBOW Wiki), FastText Arabic (300d)
   - Architecture Variations: BiLSTM, BiLSTM + CRF, and BiLSTM + Attention + CRF
   - Peak Performance: **75.04% Macro F1** / **96.05% Weighted F1**

2. **Pre-trained Transformer-Based Token Classification**
   - Models evaluated: AraELECTRA, AraBERT, CAMeLBERT, XLM-RoBERTa
   - Optimization techniques: Advanced data downsampling, probabilistic rare-class boosting, Focal Loss, and heavily customized class-weight clamping.
   - Peak Performance: **76.86% Macro F1** / **95.07% Weighted F1** (AraELECTRA)

---

## 📊 Project Highlights

### Dataset & The Imbalance Challenge

- **Source Corpus**: SSAC-UNPC Arabic corpus (~750K samples processed).
- **Domain Adaptation**: Tested across domains using the Shifaa medical dataset.
- **Classes**: 7 punctuation marks (None, Comma `،`, Semicolon `؛`, Period `.`, Question mark `؟`, Exclamation `!`, Colon `:`).
- **The Imbalance Challenge**: Highly skewed class distribution (e.g., Class 5 vs Class 1 weight ratio exceeds 20,000:1 before scaling), requiring custom weight-clamping and rare-class data sampling modifications.

### Key Features

- Sequenced deep learning architectures with CRF layers for structural layout constraints.
- Subword tokenization label masking (`-100` for non-first subwords) for transformer architectures.
- Evaluation tracking over both overall sequence consistency (Weighted F1) and minority token recovery (Macro F1).

### Methods Compared (Experimental Summary)

| Model Architecture                     | Embedding / Backbone   | Key Config / Pipeline Features                                  | Peak Accuracy | Peak Weighted F1 | Peak Macro F1 |
| :------------------------------------- | :--------------------- | :-------------------------------------------------------------- | :-----------: | :--------------: | :-----------: |
| **BiLSTM**                             | FastText / AraVec      | Upsampling, CrossEntropyLoss                                    |    85.27%     |      89.05%      |    69.93%     |
| **BiLSTM + CRF**                       | AraVec `full_uni_cbow` | 3 Layers, Hidden Dim: 512, CRF Loss                             |    96.08%     |      96.08%      |    75.03%     |
| **BiLSTM + Attention + CRF**           | AraVec `full_uni_cbow` | 3 Layers, Hidden Dim: 512, Attention                            |    96.01%     |      96.05%      |  **75.04%**   |
| **Transformer (Baseline Fine-Tuning)** | CAMeLBERT Mix          | Full Fine-Tuning, Standard Class Weights                        |  **99.11%**   |    **99.32%**    |    58.99%     |
| **Transformer (Optimized Target)**     | AraELECTRA Base        | Class Weight Clamping (Max 10K), Stratified Rare-Class Sampling |    94.15%     |      95.07%      |  **76.86%**   |

---

## 📈 Key Insights & Experimental Results

### 1. The Impact of the CRF Layer

Standard BiLSTM setups using standard `CrossEntropyLoss` struggled significantly with sequence cohesion, yielding a top Macro F1 of only **69.93%**. Adding a **CRF layer** to enforce structural sequence-level transition rules instantly boosted the Macro F1 score to over **74.8%–75.0%** because it restricted unnatural punctuation bursts.

### 2. The Transformer Imbalance Trap

Initially, baseline fine-tuning on massive models like CAMeLBERT and AraELECTRA delivered deceptively high overall accuracies and Weighted F1 scores (**>99%**). However, their Macro F1 scores plummeted to as low as **40.42%** (AraBERT) because the models over-optimized entirely for the dominant "No Punctuation" token.

### 3. Cracking the Rare Classes (The Winning Configuration)

The absolute top-performing design for resolving minority punctuation marks safely was **AraELECTRA (Step 15)**. By implementing:

- Strict class weight clamping adjustments (clamping the majority class scaling factor down from 65,949 to 10,000).
- Target probabilistic downsampling for dominant tokens.
- Validation subset filtering using rare-class boosting.

We unlocked a project-high **Macro F1 score of 76.86%**, allowing the model to cleanly recover highly sparse indicators like exclamation marks, colons, and semicolons.

---

## 🔧 Technologies & Environment

- **Deep Learning Frameworks**: PyTorch, PyTorch-CRF
- **Transformer Pipeline**: HuggingFace Transformers, Token Classification API
- **Arabic NLP Specific Tools**: camel-tools, AraVec, FastText Arabic
- **Data Infrastructure**: HuggingFace Datasets library, Pandas, NumPy
- **Visualization & Logging**: Matplotlib, Seaborn

---

## 🎓 Project Specifications & Architecture Notes

- **Scale**: Models trained using a vocabulary size of over 227K+ text features.
- **Sequence Configuration**: Static embedding pipelines processed deep contextual steps using a max sequence length of 800 tokens, while Transformer Token Classification contexts utilized localized sliding windows (max length 256–512, stride 64–128).
- **Context**: Developed and maintained as academic coursework for the NLP Program at Damascus University.

## 👤 Authors

|                                                Avatar                                                | Name                 | GitHub                | Role / Contribution |
| :--------------------------------------------------------------------------------------------------: | :------------------- | :-------------------- | :------------------ |
| <img src="https://github.com/AbdelazizAushar.png" width="45" height="45" style="border-radius:50%">  | **Abdelaziz Aushar** | @your_github_username | Core Developer      |
|   <img src="https://github.com/HamzaTinawi8.png" width="45" height="45" style="border-radius:50%">   | **Hamza Tinawi**     | @HamzaTinawi8         | Core Developer      |
| <img src="https://github.com/MahmoudMMohammed.png" width="45" height="45" style="border-radius:50%"> | **Mahmoud Mohammad** | @MahmoudMMohammed     | Core Developer      |
|    <img src="https://github.com/saranajati.png" width="45" height="45" style="border-radius:50%">    | **Sara Najati**      | @saranajati           | Core Developer      |
|    <img src="https://github.com/Tima-Dawwa.png" width="45" height="45" style="border-radius:50%">    | **Tima Dawwa**       | @Tima-Dawwa           | Core Developer      |

Created as coursework for NLP class in Damascus University
December 2025
