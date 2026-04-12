# AD-sex Minimal Framework

This directory is a cleaned, path-portable version of the AD-sex framework for GitHub release.

## Directory layout

```text
AD-sex-minimal/
  analysis/                 # stepwise preprocessing / analysis scripts
  misc/                     # shared feature-encoding helpers
  models/                   # PyTorch model definitions
  utils/                    # model configs and metrics
  outputs/                  # generated automatically
  figures/                  # generated automatically
  project_paths.py          # central path configuration
```

## Data access

This framework has two usage modes.

### 1. Full restricted pipeline

The first two steps require access to restricted raw EHR tables:

- `DEMOGRAPHIC.csv`
- `DIAGNOSIS.csv`
- `DEATH.csv`
- `ENCOUNTER.csv`
- `DISPENSING.csv`
- `PRESCRIBING.csv`

Raw data is not included in this repository. Users must obtain access separately and then point the framework to the external data directory with:

- `ADSEX_RAW_DATA_DIR`

### 2. Public reproducible pipeline

If restricted raw data cannot be shared, users can start from the processed step-1 artifacts:

- `step1_data_subseq.npz`
- `step1_data_subseq_3Darray.npz`

These should be placed under:

- `outputs/step1_construct_temporal_trajectory/<cohort>/`

Then the framework can run from training onward.

## Pipeline order

Run from the `AD-sex-minimal/` directory.

### Full pipeline

1. `python3 analysis/step0_1FL_AD_cohort_filtering.py`
2. `python3 analysis/step1_data_preprocessing.py`
3. Train one or more models:
   - `python3 model_train_lstmauto.py`
   - `python3 model_train_transauto.py`
   - `python3 model_train_gruauto.py`
   - `python3 model_train_mlpauto.py`
4. `python3 analysis/step3_1_cluster_generation_torch.py`
5. `python3 analysis/step3_2_cluster_pvalue_bonferroni.py`
6. `python3 analysis/step4_1_subtype_generation.py`
7. `python3 analysis/step4_2_subtype_analysis.py`
8. `python3 analysis/step5_1_survival_analysis.py`
9. `python3 analysis/step6_hyperparameter_sensitivity.py`

### Public pipeline

Start from step 3 if step-1 artifacts are already available.

## Outputs

All generated files are written under `outputs/`, grouped by stage and cohort.

Important outputs include:

- `outputs/step1_construct_temporal_trajectory/<cohort>/step1_data_subseq_3Darray.npz`
- `outputs/step2_training/<cohort>/torch_models/.../*.pt`
- `outputs/step3_cluster_and_interpretability/<cohort>/step3_2_ad_res.csv`
- `outputs/step4_subtype_and_interpretability/<cohort>/step4_1_subtypes.csv`

## Configuration

The framework reads these optional environment variables:

- `ADSEX_COHORT`
- `ADSEX_RAW_DATA_DIR`
- `ADSEX_OUTPUT_DIR`
- `ADSEX_FIGURES_DIR`
- `ADSEX_MAPPING_DIR`

If not set, project-relative defaults are used where possible.
