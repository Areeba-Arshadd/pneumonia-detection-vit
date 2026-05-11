 Efficient Fine-Tuning of Vision Transformer for Pneumonia Detection
 Overview

This project presents a deep learning pipeline for automated pneumonia detection from chest X-ray images using a Vision Transformer (ViT-Base-Patch16-224). The study investigates the impact of partial fine-tuning, layer freezing, and label smoothing to improve model stability and generalization on small medical datasets.

 Objective

To improve the efficiency and stability of Vision Transformers for medical image classification by:

Preventing training collapse in full fine-tuning
Reducing overfitting on small datasets
Improving generalization through partial parameter updates
Key Idea

Instead of fine-tuning all 86M parameters of ViT, we:

Freeze early encoder layers (general features)
Train only last layers (task-specific learning)
Apply label smoothing for regularization
Fix validation data leakage issues

This leads to:
✔ More stable training
✔ Better generalization
✔ Reduced overfitting

 Methodology
🔹 Model Architecture
Vision Transformer (ViT-Base-Patch16-224)
12 encoder layers
768-dimensional embeddings
Multi-head self-attention mechanism
🔹 Phase 1: Baseline Reproduction
Full fine-tuning (86M parameters)
Strong performance but:
Training collapse observed
Overfitting on small dataset
🔹 Phase 2: Proposed Improvements
✔ Layer Freezing
Frozen:
Patch embedding layer
First 10/12 encoder blocks
Trainable:
Last 2 encoder blocks
Classification head

📉 Trainable parameters reduced from 86M → 14M

✔ Label Smoothing
ε = 0.1
Prevents overconfident predictions
Improves generalization
✔ Validation Fix
Fixed data leakage in validation pipeline
Ensured clean, deterministic evaluation
🧪 Datasets
🏥 Kermany Chest X-Ray Dataset
5,863 images
Classes: Normal, Pneumonia
Strong class imbalance handled via weighting
🧬 BloodMNIST (Extension Study)
17,092 images
8-class blood cell classification
Used to test generalization of approach
📊 Results Summary
🩻 Kermany Dataset
Accuracy: ~89.9%
Recall: 94.87% (improved)
AUC-ROC: ~0.95
Key improvement: higher sensitivity for pneumonia detection
🧬 BloodMNIST
Accuracy: 90.30%
Macro F1: 89.25%
AUC-ROC: 0.9917
📈 Key Findings
Full fine-tuning leads to memorization and collapse
Partial freezing significantly stabilizes training
Recall improves at cost of slight specificity drop (clinically acceptable)
Method generalizes well across medical imaging tasks

Tech Stack
Python
PyTorch
Hugging Face Transformers
NumPy, Pandas
Matplotlib, Seaborn
Google Colab (GPU: Tesla T4)

How to Run
git clone https://github.com/Areeba-Arshadd/pneumonia-vit.git
cd pneumonia-vit

pip install -r requirements.txt

# Run training notebook or script
Key Contributions
Prevented ViT training collapse via partial fine-tuning
Reduced trainable parameters by 84%
Introduced label smoothing for medical image stability
Fixed validation leakage issue
Demonstrated cross-dataset generalization (BloodMNIST)

Results Visualization
Training curves (Phase 2 vs Phase 3)
Confusion matrices
ROC curves

BS Data Science, FAST NUCES Islamabad

Future Work
Attention visualization (clinical interpretability)
Freeze-depth ablation study
Lightweight ViT variants for low-resolution medical images
Deployment as clinical decision support tool
