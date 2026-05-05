"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: final_training.py
Description:
    Final training script for the CATS breast cancer subtype classification
    project. Trains machine learning models on the preprocessed data and evaluates
    their performance.

Final model pipeline:
    Stage 1:
    - Kruskal-Wallis feature selection (k=5)
    - Random Forest classifier
    - Task: HER2+ vs rest

    Stage 2:
    - ElasticNet feature selection (top 50)
    - Nearest Centroid classifier
    - Task: HR+ vs Triple Neg

    Final prediction:
    - Stage 1 identifies HER2+
    - Stage 2 classifies remaining samples
    - Ensemble of 15 models averaged for robustness

Usage:
    python3 code/final_training.py

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, matplotlib, rich.
"""
# %%
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "model"
# %%
from utils.cv_io import load_cv_data
from utils.cv_config import build_pipeline, build_stage2_pipeline
from sklearn.preprocessing import LabelEncoder
# %%
# Load merged training and validation data
train_merged = RESULTS_DIR / "2026-04-25_final_hierarchical/preprocessing/data/train_merged.tsv"
train_clinical = DATA_DIR / "Train_clinical.tsv"
val_call = RESULTS_DIR /"2026-04-25_final_hierarchical/preprocessing/data/validation_merged.tsv"

# %%
X_train, y, _, feature_names, _ = load_cv_data(train_merged, train_clinical)
X_train = pd.DataFrame(X_train, columns=feature_names)
X_val = pd.read_csv(val_call, sep="\t")
X_val_features = X_val.drop(columns=["Chromosome", "Start", "End", "Nclone"])
X_val = X_val_features.T
# %%
# Stage 1: HER2+ vs non-HER2+
# Binary classification:
#   1 = HER2+
#   0 = HR+ or Triple Neg
y_s1 = (y == 0).astype(int)  
stage1_pipe = build_pipeline("kw_rf", random_state=42)
stage1_pipe.set_params(
    selector__k=5,           
    clf__max_features=None   
)

stage1_pipe.fit(X_train, y_s1)
proba = stage1_pipe.predict_proba(X_val)[:, 1]
pred_s1 = (proba > 0.5).astype(int)  
print("Pred HER2+ count:", pred_s1.sum())
# %%
# Stage 2: HR+ vs Triple Neg
# Only applied to non-HER2+ samples
mask_train_s2 = (y != 0)
X_s2 = X_train[mask_train_s2]
y_s2 = y[mask_train_s2]  
from sklearn.preprocessing import LabelEncoder
le_s2 = LabelEncoder()
y_s2_enc = le_s2.fit_transform(y_s2)
print(le_s2.classes_)

# %%
# Plateau ensemble (15 models)
# Each model uses ElasticNet feature selection + NMC classifier
# Predictions will be averaged across all models
configs = [
    {"C":0.0464159,"l1_ratio":0.1,"shrink_threshold":None},
    {"C":0.0464159,"l1_ratio":0.1,"shrink_threshold":0.2},
    {"C":0.0464159,"l1_ratio":0.1,"shrink_threshold":0.1},
    {"C":0.16681,"l1_ratio":0.5,"shrink_threshold":0.2},
    {"C":0.16681,"l1_ratio":0.5,"shrink_threshold":0.1},
    {"C":0.16681,"l1_ratio":0.5,"shrink_threshold":0.5},
    {"C":0.16681,"l1_ratio":0.5,"shrink_threshold":None},
    {"C":0.0464159,"l1_ratio":0.1,"shrink_threshold":0.5},
    {"C":0.599484,"l1_ratio":0.7,"shrink_threshold":0.2},
    {"C":0.16681,"l1_ratio":0.3,"shrink_threshold":0.1},
    {"C":0.16681,"l1_ratio":0.3,"shrink_threshold":0.2},
    {"C":0.16681,"l1_ratio":0.1,"shrink_threshold":None},
    {"C":0.16681,"l1_ratio":0.1,"shrink_threshold":0.2},
    {"C":0.16681,"l1_ratio":0.1,"shrink_threshold":0.1},
    {"C":0.599484,"l1_ratio":0.7,"shrink_threshold":0.1},
]
# %%
stage2_models = []
for cfg in configs:
    pipe = build_stage2_pipeline("en_nmc", random_state=42)
    
    pipe.set_params(
        selector__top_k=50,
        selector__C=cfg["C"],
        selector__l1_ratio=cfg["l1_ratio"],
        clf__shrink_threshold=cfg["shrink_threshold"]
    )
    
    pipe.fit(X_s2, y_s2_enc)
    stage2_models.append(pipe)
# %%
mask_val_s2 = (pred_s1 == 0)
X_val_s2 = X_val[mask_val_s2]
# %%
probs = []
for model in stage2_models:
    probs.append(model.predict_proba(X_val_s2))

avg_proba = np.mean(probs, axis=0)
pred_s2 = np.argmax(avg_proba, axis=1)
# %%
# Combine Stage 1 and Stage 2 predictions
# Stage 1 decides HER2+
# Remaining samples classified by Stage 2
final_pred = []

idx_s2 = 0

for i in range(len(pred_s1)):
    if pred_s1[i] == 1:
        final_pred.append("HER2+")
    else:
        label = le_s2.inverse_transform([pred_s2[idx_s2]])[0]
        
        if label == 1:
            final_pred.append("HR+")
        else:
            final_pred.append("Triple Neg")
        
        idx_s2 += 1
# %%
sample_names = X_val.index

pred_df = pd.DataFrame({
    "Sample": sample_names,
    "Subgroup": final_pred
})
# Save predictions
pred_df.to_csv(
    "prediction.txt",
    sep="\t",
    index=False,
    header=['"Sample"', '"Subgroup"']
)