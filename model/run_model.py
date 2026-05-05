import sys
import argparse
import joblib
import pandas as pd
import numpy as np

from utils.cv_components import KruskalWallisSelector, ElasticNetSelector, NearestCentroidWithProba


def build_consensus_matrix(cn_df, merge_map, sample_cols):
    data = cn_df[sample_cols].values.astype(float)
    n_segments = len(merge_map)

    cn_matrix = np.empty((n_segments, len(sample_cols)))

    for seg_id_str in merge_map:
        seg_id = int(seg_id_str)
        raw_indices = merge_map[seg_id_str]
        cn_matrix[seg_id] = np.median(data[raw_indices], axis=0)

    return pd.DataFrame(cn_matrix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-m", required=True)
    parser.add_argument("-o", required=True)
    args = parser.parse_args()

    # ===== load model =====
    model = joblib.load(args.m)

    stage1 = model["stage1_pipeline"]
    stage2_models = model["stage2_pipelines"]
    le_s2 = model["label_encoder_stage2"]
    feature_names = model["feature_names"]
    merge_map = model["merge_map"]

    # ===== load raw validation data =====
    val_df = pd.read_csv(args.i, sep="\t")
    sample_cols = [c for c in val_df.columns if c.startswith("Array")]

    # ===== merge 2834 → 273 =====
    val_merged = build_consensus_matrix(val_df, merge_map, sample_cols)

    X_val = val_merged.T
    X_val.columns = feature_names
    X_val.index = sample_cols
    X_val = X_val[feature_names]

    # ===== Stage 1: HER2+ vs rest =====
    proba = stage1.predict_proba(X_val)[:, 1]
    pred_s1 = (proba > 0.5).astype(int)

    # ===== Stage 2: HR+ vs TN =====
    mask = (pred_s1 == 0)
    X_val_s2 = X_val[mask]

    probs = [m.predict_proba(X_val_s2) for m in stage2_models]
    avg_proba = np.mean(probs, axis=0)
    pred_s2 = np.argmax(avg_proba, axis=1)

    # ===== combine predictions =====
    final_pred = []
    idx = 0

    for i in range(len(pred_s1)):
        if pred_s1[i] == 1:
            final_pred.append("HER2+")
        else:
            label = le_s2.inverse_transform([pred_s2[idx]])[0]
            final_pred.append("HR+" if label == 1 else "Triple Neg")
            idx += 1

    # ===== save output =====
    out_df = pd.DataFrame({
        "Sample": X_val.index,
        "Subgroup": final_pred
    })

    out_df.columns = ['"Sample"', '"Subgroup"']
    out_df.to_csv(args.o, sep="\t", index=False)


if __name__ == "__main__":
    main()