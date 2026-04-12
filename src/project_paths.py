from __future__ import annotations

import os
from pathlib import Path


FRAMEWORK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FRAMEWORK_ROOT.parents[1]

COHORT = os.getenv("ADSEX_COHORT", "2022_1FL")

RAW_DATA_DIR = Path(os.getenv("ADSEX_RAW_DATA_DIR", FRAMEWORK_ROOT / "raw_data"))
OUTPUT_ROOT = Path(os.getenv("ADSEX_OUTPUT_DIR", FRAMEWORK_ROOT / "outputs"))
FIGURES_DIR = Path(os.getenv("ADSEX_FIGURES_DIR", FRAMEWORK_ROOT / "figures"))
MAPPING_DIR = Path(os.getenv("ADSEX_MAPPING_DIR", REPO_ROOT / "mapping"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_dir(stage_name: str, cohort: str = COHORT) -> Path:
    return ensure_dir(OUTPUT_ROOT / stage_name / cohort)


STEP0_DIR = stage_dir("step0_data_preprocessing")
STEP1_DIR = stage_dir("step1_construct_temporal_trajectory")
STEP2_DIR = stage_dir("step2_training")
STEP3_DIR = stage_dir("step3_cluster_and_interpretability")
STEP4_DIR = stage_dir("step4_subtype_and_interpretability")

TORCH_MODEL_DIR = ensure_dir(STEP2_DIR / "torch_models")

STEP0_DEMO_CSV = STEP0_DIR / "step0_demo_ad.csv"
STEP1_PATIENT_PKL = STEP1_DIR / "step1_data.pkl"
STEP1_SUBSEQ_NPZ = STEP1_DIR / "step1_data_subseq.npz"
STEP1_3D_NPZ = STEP1_DIR / "step1_data_subseq_3Darray.npz"

STEP3_LATENT_NPZ = STEP3_DIR / "step3_1_latent_features.npz"
STEP3_CLUSTER_CSV = STEP3_DIR / "step3_2_ad_res.csv"
STEP3_PAIRWISE_CHISQ_CSV = STEP3_DIR / "step3_2_pairwise_chisq.csv"

STEP4_SUBTYPE_CSV = STEP4_DIR / "step4_1_subtypes.csv"
STEP4_CLUSTER_SUBTYPE_CSV = STEP4_DIR / "step4_2_cluster_subtype.csv"
STEP4_SUBTYPE_GROUP_OR_CSV = STEP4_DIR / "step4_2_subtype_group_odds_ratio.csv"
STEP4_SUBTYPE_PAIR_OR_CSV = STEP4_DIR / "step4_2_subtype_pair_odds_ratio.csv"
STEP4_SUBTYPE_IMPORTANCE_METRICS_CSV = STEP4_DIR / "step4_2_subtype_feature_importance_metrics.csv"
STEP4_SUBTYPE_IMPORTANCE_CSV = STEP4_DIR / "step4_2_subtype_feature_importance.csv"
STEP6_SENSITIVITY_CSV = STEP3_DIR / "step6_sensitivity_results.csv"

PCA_FIG_PATH = FIGURES_DIR / "step3_pca_of_clusters_with_labels.svg"


def raw_table(name: str) -> Path:
    return RAW_DATA_DIR / name


def mapping_file(name: str) -> Path:
    return MAPPING_DIR / name


def model_save_dir(model_name: str, data_sources: list[str], month: int, layers: list[int], seed: int) -> Path:
    layer_info = "-".join(map(str, layers))
    source_info = "-".join(data_sources)
    return ensure_dir(
        TORCH_MODEL_DIR / f"model_{model_name}" / f"source{source_info}_month{month}_layer{layer_info}_seed{seed}"
    )
