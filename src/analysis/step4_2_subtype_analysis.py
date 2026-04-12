import warnings
from itertools import combinations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.contingency_tables import Table2x2
from tqdm import tqdm
from project_paths import (
    STEP3_CLUSTER_CSV,
    STEP4_SUBTYPE_CSV,
    STEP4_DIR,
    STEP4_CLUSTER_SUBTYPE_CSV,
    STEP4_SUBTYPE_GROUP_OR_CSV,
    STEP4_SUBTYPE_PAIR_OR_CSV,
    STEP4_SUBTYPE_IMPORTANCE_METRICS_CSV,
    STEP4_SUBTYPE_IMPORTANCE_CSV,
    ensure_dir,
)

warnings.filterwarnings("ignore")

CLUSTER_CSV = STEP3_CLUSTER_CSV
SUBTYPE_CSV = STEP4_SUBTYPE_CSV
OUTPUT_DIR = STEP4_DIR

SUBTYPES = ["[0 1 2 3]", "[0 2 3]", "[0 4 5 6]", "[0 5 6]", "[0 4 5]"]


def group_odds_ratio(df):
    results = []
    groups = df["cls_pattern_encoded"].unique()
    feature_columns = [col for col in df.columns if col not in ["cls_pattern_encoded"]]

    group_sums = df.groupby("cls_pattern_encoded")[feature_columns].sum()
    group_sizes = df.groupby("cls_pattern_encoded").size()
    total_sums = df[feature_columns].sum()
    total_size = len(df)

    for subtype_id in tqdm(groups, desc="Computing subtype odds ratios"):
        group_sum = group_sums.loc[subtype_id]
        group_size = group_sizes.loc[subtype_id]

        baseline_sum = total_sums - group_sum
        baseline_size = total_size - group_size

        odds_in_group = group_sum / (group_size - group_sum + 1e-10)
        odds_in_baseline = baseline_sum / (baseline_size - baseline_sum + 1e-10)
        or_values = odds_in_group / odds_in_baseline

        for feature in feature_columns:
            table = [
                [group_sum[feature], baseline_sum[feature]],
                [group_size - group_sum[feature], baseline_size - baseline_sum[feature]],
            ]
            analysis = Table2x2(table)

            results.append(
                {
                    "feature": feature,
                    "subtype": subtype_id,
                    "odds_ratio": or_values[feature],
                    "p_value": analysis.oddsratio_pvalue(),
                    "ci_lower": analysis.oddsratio_confint()[0],
                    "ci_upper": analysis.oddsratio_confint()[1],
                }
            )

    return pd.DataFrame(results)


def pair_odds_ratio(df):
    results = []
    groups = df["cls_pattern_encoded"].unique()
    feature_columns = [col for col in df.columns if col not in ["cls_pattern_encoded"]]

    group_sums = df.groupby("cls_pattern_encoded")[feature_columns].sum()
    group_sizes = df.groupby("cls_pattern_encoded").size()

    for subtype_a, subtype_b in tqdm(combinations(groups, 2), desc="Computing subtype pairs"):
        sum_a = group_sums.loc[subtype_a]
        size_a = group_sizes.loc[subtype_a]
        sum_b = group_sums.loc[subtype_b]
        size_b = group_sizes.loc[subtype_b]

        odds_a = sum_a / (size_a - sum_a + 1e-10)
        odds_b = sum_b / (size_b - sum_b + 1e-10)
        or_values = odds_a / odds_b

        for feature in feature_columns:
            table = [
                [sum_a[feature], sum_b[feature]],
                [size_a - sum_a[feature], size_b - sum_b[feature]],
            ]
            analysis = Table2x2(table)

            results.append(
                {
                    "feature": feature,
                    "subtype_a": subtype_a,
                    "subtype_b": subtype_b,
                    "odds_ratio": or_values[feature],
                    "p_value": analysis.oddsratio_pvalue(),
                    "ci_lower": analysis.oddsratio_confint()[0],
                    "ci_upper": analysis.oddsratio_confint()[1],
                }
            )

    return pd.DataFrame(results)


def extract_significant_results(df, group_col):
    results = {}
    for group in sorted(df[group_col].unique()):
        tmp = df[df[group_col] == group].copy()
        tmp = tmp[tmp["p_value"] < 0.05]
        tmp = tmp[tmp["odds_ratio"].notna()]
        tmp = tmp[~tmp["odds_ratio"].isin([float("inf"), float("-inf")])]
        tmp = tmp.sort_values(by="odds_ratio", ascending=False)
        results[group] = tmp
    return results


def split_feature_types(df):
    phe_df = df[df["feature"].str.startswith("Phe_")].copy()
    atc_df = df[df["feature"].str.startswith("ATC_")].copy()
    demo_df = df[
        (~df["feature"].str.startswith("Phe_")) & (~df["feature"].str.startswith("ATC_"))
    ].copy()
    return demo_df, phe_df, atc_df


def build_cluster_subtype_table(df_cluster, df_subtypes):
    df_cluster_subtype = pd.merge(
        df_cluster, df_subtypes[["PATID", "cls_pattern"]], on="PATID", how="left"
    )

    cols = list(df_cluster_subtype.columns)
    cols = cols[:4] + [cols[-1]] + cols[4:-1]
    df_cluster_subtype = df_cluster_subtype[cols]

    df_cluster_subtype = df_cluster_subtype[
        df_cluster_subtype["cls_pattern"].isin(SUBTYPES)
    ].reset_index(drop=True)

    encoder = LabelEncoder()
    df_cluster_subtype["cls_pattern_encoded"] = encoder.fit_transform(
        df_cluster_subtype["cls_pattern"]
    )

    cols = list(df_cluster_subtype.columns)
    cols = cols[:5] + [cols[-1]] + cols[5:-1]
    return df_cluster_subtype[cols]


def subtype_feature_importance(df_cluster_subtype):
    x = df_cluster_subtype.filter(like="Phe_")
    y = df_cluster_subtype["cls_pattern_encoded"].apply(lambda value: 1 if value in [0, 1] else 0)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics_df = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
        ]
    )

    importance_df = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.coef_[0]}
    )
    importance_df["feature"] = importance_df["feature"].str.replace("Phe_", "", regex=False)
    importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    return metrics_df, importance_df


def save_subtype_or_outputs(df_cluster_subtype, df_subtype_or, output_dir):
    group_results = extract_significant_results(df_subtype_or, "subtype")
    pair_results = pair_odds_ratio(df_cluster_subtype.iloc[:, 5:])

    df_subtype_or.to_csv(STEP4_SUBTYPE_GROUP_OR_CSV, index=False)
    pair_results.to_csv(STEP4_SUBTYPE_PAIR_OR_CSV, index=False)

    phe_output_dir = ensure_dir(output_dir / "group_OR_subtype")

    for subtype_id, df in group_results.items():
        _, phe_df, atc_df = split_feature_types(df)
        phe_df.to_csv(
            phe_output_dir / f"step4_2_subtype_{subtype_id}_phe.csv",
            index=False,
        )
        atc_df.to_csv(
            phe_output_dir / f"step4_2_subtype_{subtype_id}_atc.csv",
            index=False,
        )


if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    df_cluster = pd.read_csv(CLUSTER_CSV)
    df_subtypes = pd.read_csv(SUBTYPE_CSV)

    df_cluster_subtype = build_cluster_subtype_table(df_cluster, df_subtypes)
    df_cluster_subtype.to_csv(STEP4_CLUSTER_SUBTYPE_CSV, index=False)

    df_subtype_or = group_odds_ratio(df_cluster_subtype.iloc[:, 5:])
    save_subtype_or_outputs(df_cluster_subtype, df_subtype_or, OUTPUT_DIR)

    metrics_df, importance_df = subtype_feature_importance(df_cluster_subtype)
    metrics_df.to_csv(
        STEP4_SUBTYPE_IMPORTANCE_METRICS_CSV,
        index=False,
    )
    importance_df.to_csv(
        STEP4_SUBTYPE_IMPORTANCE_CSV,
        index=False,
    )
