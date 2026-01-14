# Physics-Guided-Causal-Diffusion-with-Confidence-Rectified-Ensemble

PGCD-CRE: Physics-Guided Causal Diffusion for GZSL Fault DiagnosisThis repository contains the official PyTorch implementation of the paper:Physics-Guided Causal Diffusion with Confidence-Rectified Ensemble for Generalized Zero-Shot Compound Fault Diagnosis Submitted to IEEE Transactions on [Journal Name]📝 AbstractGeneralized Zero-Shot Learning (GZSL) in machinery fault diagnosis aims to identify novel compound faults (unseen) using only single-fault data (seen). We propose PGCD-CRE, a framework that bridges the semantic gap through physical causality.Key innovations:Physics-Guided Causal Diffusion: Generates high-fidelity unseen compound fault signals by enforcing physical envelope constraints and multi-hot causal conditioning.Specialist Ensemble Diagnosis: A diagnostic strategy with Cross-Confidence Rectification to resolve spectral confusion between similar compound faults (e.g., Inner-Ball vs. Outer-Ball).(Note: You should upload your framework figure to an 'assets' folder)📂 Project StructureBash├── dataset/                  # Dataset directory
│   ├── xjtu/                 # XJTU Gearbox Dataset files
│   └── hust/                 # HUST Bearing Dataset files
├── models/                   # Model definitions
│   ├── causal_diffusion_model.py  # Causal UNet & Physics-Guided Diffusion
│   └── classifier.py         # Specialist & Unified Classifiers
├── utils/                    # Utility functions
│   ├── data_loader.py        # Custom Dataset classes
│   └── physics_loss.py       # Envelope Spectrum Loss
├── experiments/              # Ablation & Analysis Scripts
│   ├── ablation_study_xjtu.py            # Scheme A: Generation Analysis
│   ├── ablation_study_hust_classifier.py # Scheme B: Diagnostic Strategy
│   └── ablation_study_sensitivity.py     # Scheme C: Parameter Sensitivity
├── main.py                   # Main entry point for full pipeline
├── requirements.txt          # Dependencies
└── README.md
🛠️ PrerequisitesLinux or WindowsPython 3.8+PyTorch 1.10+NVIDIA GPU (Recommended)Install dependencies:Bashpip install -r requirements.txt
Key requirements: torch, numpy, scipy, sklearn, tqdm, matplotlib, seaborn.💾 Data Preparation1. XJTU Gearbox DatasetDownload the dataset from the XJTU Website (or your source).Organize files as follows:dataset/xjtu/
├── Ball.mat
├── Inner.mat
├── Outer.mat
└── Mix.mat
2. HUST Bearing DatasetDownload the dataset from [HUST source].Organize files as follows:dataset/hust/
├── N504.mat
├── B504.mat
├── I504.mat
├── O504.mat
├── IB504.mat
└── OB504.mat
🚀 Usage1. Training the Diffusion ModelTo train the Physics-Guided Causal Diffusion model on seen classes:Bashpython main.py --mode train_diffusion --dataset xjtu --epochs 150
2. Zero-Shot GenerationTo synthesize unseen compound fault signals (e.g., Mix fault):Bashpython main.py --mode generate --dataset xjtu --n_samples 500 --guidance_scale 10.0
3. GZSL Diagnosis (Training & Evaluation)To train the classifier on the mixture of Real Seen + Synthetic Unseen data:Bashpython main.py --mode gzsl_eval --dataset xjtu
📊 Reproduction of Ablation StudiesWe provide specific scripts to reproduce the results reported in the paper's Discussion section.Scheme A: Impact of Generative Mechanisms (XJTU)Compares Vanilla DDPM, Causal DDPM, and Proposed PGCD.Bashpython experiments/ablation_study_xjtu.py
Scheme B: Diagnostic Strategy Evolution (HUST)Compares Unified Classifier, Simple Ensemble, and Proposed PGCD-CRE.Bashpython experiments/ablation_study_hust_classifier.py
Scheme C: Sensitivity Analysis of Guidance ScaleAnalyzes the impact of $s$ on performance (Inverted U-shape).Bashpython experiments/ablation_study_sensitivity.py
📈 Main ResultsXJTU Dataset (Generation Quality)MethodSeen AccUnseen AccH-scoreVanilla DDPM99.78%31.27%47.61%Causal DDPM99.78%44.54%61.59%Proposed99.93%88.42%93.82%HUST Dataset (Diagnostic Strategy)StrategySeen AccUnseen AccH-scoreUnified Classifier98.87%42.29%59.44%Simple Ensemble98.87%25.98%41.24%Proposed (CRE)99.38%96.00%97.66%🔗 CitationIf you find this code useful for your research, please cite our paper:代码段@article{YourName2026PGCD,
  title={Physics-Guided Causal Diffusion with Confidence-Rectified Ensemble for Generalized Zero-Shot Compound Fault Diagnosis},
  author={Your Name and Co-authors},
  journal={IEEE Transactions on ...},
  year={2026}
}
📧 ContactFor any questions, please open an issue or contact:Author Name: [email@example.com]
