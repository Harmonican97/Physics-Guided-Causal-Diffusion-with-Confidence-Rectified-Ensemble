# Physics-Guided-Causal-Diffusion-with-Confidence-Rectified-Ensemble

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![GZSL](https://img.shields.io/badge/Task-Generalized%20Zero--Shot%20Learning-green.svg)]()

Official PyTorch implementation of the paper: 
**Physics-Guided Causal Diffusion with Confidence-Rectified Ensemble for Generalized Zero-Shot Compound Fault Diagnosis** *Submitted to IEEE Transactions on Industrial Informatics*

## 📝 Abstract

Generalized Zero-Shot Learning (GZSL) in machinery fault diagnosis aims to identify novel compound faults (unseen) using only single-fault data (seen). Traditional generative methods often fail to capture the physical coupling of fault signatures, leading to poor generalization.

We propose **PGCD-CRE**, a novel framework that bridges the semantic gap through physical causality.
Key innovations include:
1.  **Physics-Guided Causal Diffusion:** Generates high-fidelity unseen compound fault signals by enforcing physical envelope constraints and multi-hot causal conditioning.
2.  **Specialist Ensemble Diagnosis:** A diagnostic strategy with Cross-Confidence Rectification to resolve spectral confusion between similar compound faults (e.g., Inner-Ball vs. Outer-Ball).

![Framework](assets/framework.png)

## 📖 Overview
This repository contains the code for the **PGCD-CRE** framework, a novel generative approach for Generalized Zero-Shot Learning (GZSL) in industrial rotating machinery. The framework synthesizes high-fidelity compound fault signals using a **Physics-Guided Causal Diffusion Model (PGCD)** and employs a **Cross-Confidence Rectification Ensemble (CRE)** to mitigate domain shifts during diagnosis.

## 📂 Repository Structure
```text
.
├── dataset/
│   ├── xjtu/                   # Place XJTU planetary gearbox dataset here
│   └── hust/                   # Place HUST bearing dataset here
├── main_gzsl_xjtu.py           # End-to-end training & evaluation for XJTU dataset
├── main_gzsl_hust.py           # PGCD generation phase for HUST dataset
├── hust_gzsl_sp_model.py       # Specialist model training for HUST dataset
├── hust_gzsl_sp_model_confidence.py # Final evaluation with CRE strategy for HUST
└── README.md                   # This documentation

```

## 💾 Data Preparation

### 1. XJTU Gearbox Dataset
Download the dataset from the XJTU Website and organize files as follows:

```bash
dataset/xjtu/
├── 1ndBearing_ball/ 
│   └── Data_Chan1.txt
├── 1ndBearing_inner/ 
│   └── Data_Chan1.txt
├── 1ndBearing_outer/ 
│   └── Data_Chan1.txt
└── 1ndBearing_mix(inner+outer+ball)/ 
    └── Data_Chan1.txt
```
### 2. HUST Bearing Dataset
Download the dataset and organize files as follows:

```bash
dataset/hust/
├── N504.mat
├── B504.mat
├── I504.mat
├── O504.mat
├── IB504.mat
└── OB504.mat
```

## ⚙️ Requirements

To install the required dependencies, run the following command or ensure your environment has these packages installed:

```bash
pip install torch torchvision torchaudio numpy scipy scikit-learn matplotlib seaborn tqdm
```

## 🚀 Usage

### 1. XJTU Dataset Experiment

To train the PGCD model and the robust classifier for the complex compound fault diagnosis (e.g., Inner + Outer + Ball) on the XJTU dataset, run:

```bash
python main_gzsl_xjtu.py
```

**Expected Pipeline:**

1. Loads the seen single-fault data from `./dataset/xjtu`.
2. Trains the physics-guided diffusion model (`causal_diffusion_model.py`).
3. Generates refined synthetic compound fault samples using explicit kinematic constraints.
4. Trains the robust classifier on the mixed dataset (Real Seen + Synthetic Unseen).
5. **Outputs:** Prints `Seen Acc`, `Unseen Acc`, and `H-score` to the console and generates the confusion matrix plot.

---

### 2. HUST Dataset Experiment

The HUST bearing dataset evaluation involves a three-step pipeline designed to validate the **Cross-Confidence Rectified Ensemble (CRE)** strategy for multiple unseen compound faults.

**Step 2.1: Train PGCD and Generate Synthetic Data**
First, train the diffusion model to synthesize high-fidelity compound fault data for the unseen classes:

```bash
python main_gzsl_hust.py
```

*(This script saves the generated synthetic tensors to the `./results_gzsl_final` directory.)*

**Step 2.2: Train the Specialist Models**
Next, train the individual specialist classifiers on their respective fault domains:

```bash
python hust_gzsl_sp_model.py
```

*(This script trains and saves the specialist model weights to the `./results_gzsl_ensemble` directory.)*

**Step 2.3: Evaluate with Confidence-Rectified Ensemble**
Finally, execute the CRE strategy to perform the global zero-shot diagnosis and rectify confidence biases:

```bash
python hust_gzsl_sp_model_confidence.py
```

**Expected Outputs:** The script will print the final `Seen Acc`, `Unseen Acc`, and `H-score` metrics. It will also generate and save the refined Confusion Matrix (`conf_cm.png`) and t-SNE visualizations in the `./results_gzsl_confidence` directory.



