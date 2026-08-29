"""
Group 9.
Authors:
    Alexandros Michailidis.
    Antonie Wagner.
    Christos Botos.
    Yan Qiao.
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: run_model.py.
Description:
    Load the serialised hierarchical classifier and predict subtypes for
    new validation samples. Handles region merging from raw aCGH input
    and applies the two-stage prediction (HER2+ vs rest, then HR+ vs TN).

Usage:
    python3 model/run_model.py -i data/Validation_call.tsv -m model/model.pkl -o output.txt

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, joblib.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Ensure code/ is on sys.path so joblib can resolve custom classes
# (KruskalWallisSelector, ElasticNetSelector, NearestCentroidWithProba)
# that were pickled with module paths under utils.cv_components.
_CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


def build_consensus_matrix(cn_df, merge_map, sample_cols):
    """Merge raw genomic regions into consensus segments using the merge map.

    Applies median aggregation to collapse groups of correlated raw regions
    into single consensus segments, matching the preprocessing used during
    training. Sorts the output by genomic position to match the row order
    produced by preprocessing_phase.py.

    Args:
        cn_df (pd.DataFrame): Raw copy-number data with genomic columns
            and sample columns.
        merge_map (dict): Mapping from segment ID (str) to list of raw
            region indices.
        sample_cols (list[str]): Sample column names to extract.

    Returns:
        pd.DataFrame: Merged consensus matrix with segments as rows and
            samples as columns, sorted by genomic position.
    """
    data = cn_df[sample_cols].values.astype(float)
    n_segments = len(merge_map)
    chrom_arr = np.empty(n_segments, dtype=int)
    start_arr = np.empty(n_segments, dtype=int)
    cn_matrix = np.empty((n_segments, len(sample_cols)))

    for seg_id_str in merge_map:
        seg_id = int(seg_id_str)
        raw_indices = merge_map[seg_id_str]
        cn_matrix[seg_id] = np.median(data[raw_indices], axis=0)
        constituent = cn_df.iloc[raw_indices]
        chrom_arr[seg_id] = int(constituent["Chromosome"].iloc[0])
        start_arr[seg_id] = int(constituent["Start"].min())

    consensus_df = pd.DataFrame(cn_matrix, columns=sample_cols)
    consensus_df.insert(0, "Chromosome", chrom_arr)
    consensus_df.insert(1, "Start", start_arr)

    # Sort by genomic position to match preprocessing_phase.py output order.
    consensus_df = consensus_df.sort_values(
        ["Chromosome", "Start"],
    ).reset_index(drop=True)

    # Drop coordinate columns, returning only sample data.
    consensus_df = consensus_df.drop(columns=["Chromosome", "Start"])

    return consensus_df


def main():
    """Load model, preprocess validation data, and generate predictions."""
    parser = argparse.ArgumentParser(
        description="Predict breast cancer subtypes from aCGH data.",
    )
    parser.add_argument("-i", required=True, help="Input validation TSV.")
    parser.add_argument("-m", required=True, help="Path to model.pkl.")
    parser.add_argument("-o", required=True, help="Output prediction file.")
    args = parser.parse_args()

    """Load Model"""

    model = joblib.load(args.m)
    stage1 = model["stage1_pipeline"]
    stage2_models = model["stage2_pipelines"]
    le_s2 = model["label_encoder_stage2"]
    feature_names = model["feature_names"]
    merge_map = model["merge_map"]

    """Load and Preprocess Validation Data"""

    val_df = pd.read_csv(args.i, sep="\t")
    genomic_cols = ("Chromosome", "Start", "End", "Nclone")
    sample_cols = [c for c in val_df.columns if c not in genomic_cols]

    # Merge raw regions into consensus segments.
    val_merged = build_consensus_matrix(val_df, merge_map, sample_cols)

    X_val = val_merged.T
    X_val.columns = feature_names
    X_val.index = sample_cols
    X_val = X_val[feature_names]

    """Stage 1: HER2+ vs Rest"""

    proba = stage1.predict_proba(X_val)[:, 1]
    pred_s1 = (proba >= 0.5).astype(int)

    """Stage 2: HR+ vs Triple Neg"""

    mask = pred_s1 == 0
    X_val_s2 = X_val[mask]

    probs = [m.predict_proba(X_val_s2) for m in stage2_models]
    avg_proba = np.mean(probs, axis=0)
    pred_s2 = np.argmax(avg_proba, axis=1)

    """Combine Predictions"""

    final_pred = []
    idx = 0
    for i in range(len(pred_s1)):
        if pred_s1[i] == 1:
            final_pred.append("HER2+")
        else:
            label = le_s2.inverse_transform([pred_s2[idx]])[0]
            if label == 1:
                final_pred.append("HR+")
            else:
                final_pred.append("Triple Neg")
            idx += 1

    """Save Output"""

    with open(args.o, "w", newline="\n") as f:
        f.write('"Sample"\t"Subgroup"\n')
        for sample, label in zip(X_val.index, final_pred):
            f.write(f"{sample}\t{label}\n")


if __name__ == "__main__":
    main()
