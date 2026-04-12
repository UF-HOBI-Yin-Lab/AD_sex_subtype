"""
Step 3-2: Pairwise cluster comparison using chi-square or Fisher's exact test.

For each binary feature and each pair of clusters, build a 2x2 contingency table:

             feature=1   feature=0
cluster_i        a           b
cluster_j        c           d

Use Fisher's exact test when any cell count is < 5; otherwise use chi-square.
Apply Bonferroni correction and FDR correction within each cluster pair across
all tested features.

Output: one row per (cluster_i, cluster_j, feature) comparison.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from project_paths import STEP3_CLUSTER_CSV, STEP3_PAIRWISE_CHISQ_CSV

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLUSTER_CSV = STEP3_CLUSTER_CSV
OUTPUT_CSV = STEP3_PAIRWISE_CHISQ_CSV

# Minimum feature prevalence to include a feature (fraction of all rows)
MIN_PREVALENCE = 0.02


def choose_test(table):
    """Return test name, p-value, and odds ratio/statistic for a 2x2 table."""
    if np.any(table < 5):
        odds_ratio, p_value = fisher_exact(table)
        return "fisher_exact", p_value, odds_ratio

    chi2, p_value, _, _ = chi2_contingency(table, correction=False)
    return "chi_square", p_value, chi2


def main():
    print("Loading cluster data ...")
    df_cluster = pd.read_csv(CLUSTER_CSV)

    # Expected columns: PATID, subseq_PATID, label, cluster, features...
    feature_cols = list(df_cluster.columns[4:])
    print(f"Loaded shape: {df_cluster.shape}")
    print(f"Clusters found: {sorted(df_cluster['cluster'].unique())}")

    prevalence = df_cluster[feature_cols].mean(axis=0)
    kept_features = prevalence[prevalence > MIN_PREVALENCE].index.tolist()
    print(f"Features before filter: {len(feature_cols)}")
    print(f"Features after filter : {len(kept_features)} (prevalence > {MIN_PREVALENCE})")

    clusters = sorted(df_cluster["cluster"].unique())
    pairs = [
        (clusters[i], clusters[j])
        for i in range(len(clusters) - 1)
        for j in range(i + 1, len(clusters))
    ]

    results = []

    for ci, cj in pairs:
        sub = df_cluster[df_cluster["cluster"].isin([ci, cj])].copy()
        print(f"Comparing cluster {ci} vs {cj} ...")

        for feature in tqdm(kept_features, desc=f"Pair {ci}-{cj}", leave=False):
            tmp = sub[["cluster", feature]].dropna()

            ci_vals = tmp[tmp["cluster"] == ci][feature].astype(int)
            cj_vals = tmp[tmp["cluster"] == cj][feature].astype(int)

            ci_1 = int((ci_vals == 1).sum())
            ci_0 = int((ci_vals == 0).sum())
            cj_1 = int((cj_vals == 1).sum())
            cj_0 = int((cj_vals == 0).sum())

            table = np.array([[ci_1, ci_0], [cj_1, cj_0]])
            test_used, raw_p, stat_value = choose_test(table)

            results.append(
                {
                    "cluster_i": ci,
                    "cluster_j": cj,
                    "feature": feature,
                    "cluster_i_feature_1": ci_1,
                    "cluster_i_feature_0": ci_0,
                    "cluster_j_feature_1": cj_1,
                    "cluster_j_feature_0": cj_0,
                    "cluster_i_prevalence": ci_1 / max(ci_1 + ci_0, 1),
                    "cluster_j_prevalence": cj_1 / max(cj_1 + cj_0, 1),
                    "test_used": test_used,
                    "stat_value": stat_value,
                    "raw_p": raw_p,
                }
            )

    df_res = pd.DataFrame(results)

    print("Applying multiple-testing correction ...")
    corrected_frames = []
    for (ci, cj), group in df_res.groupby(["cluster_i", "cluster_j"], sort=False):
        raw_p = group["raw_p"].astype(float).values

        _, bonf_p, _, _ = multipletests(raw_p, method="bonferroni")
        _, fdr_p, _, _ = multipletests(raw_p, method="fdr_bh")

        group = group.copy()
        group["p_bonferroni"] = bonf_p
        group["p_fdr_bh"] = fdr_p
        corrected_frames.append(group)

    df_out = pd.concat(corrected_frames, ignore_index=True)
    df_out = df_out.sort_values(
        by=["cluster_i", "cluster_j", "p_bonferroni", "raw_p", "feature"]
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved pairwise comparison table to:\n  {OUTPUT_CSV}")
    print(f"Output shape: {df_out.shape}")


if __name__ == "__main__":
    main()
