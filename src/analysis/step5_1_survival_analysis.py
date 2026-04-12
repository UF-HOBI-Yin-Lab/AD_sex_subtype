import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import STEP0_DEMO_CSV, STEP4_CLUSTER_SUBTYPE_CSV, FIGURES_DIR, ensure_dir

COMORBIDITY_COLS = ['Phe_Essential hypertension', 'Phe_Hyperlipidemia', 'Phe_Hypertension']
OBSERVATION_WINDOW_DAYS = 5 * 365  # right-censor at 5 years from comorbidity onset

# S0/S1: female-dominant (reference); S2/S3/S4: male-dominant
MALE_DOMINANT_SUBTYPES = {2, 3, 4}

SAVE_DIR = ensure_dir(FIGURES_DIR)

# Landmark analysis constants
LANDMARK_DAYS = 1 * 365    # landmark time: 1 year from first EHR encounter
LANDMARK_WINDOW_DAYS = 5 * 365  # follow-up window after landmark (right-censor at 5 yr)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------

def _extract_comorbidity_onset(df_cluster_subtype, col):
    """Return DataFrame with PATID and first_comorbidity_days for patients
    who ever had `col` == 1."""
    records = []
    for patid, group in df_cluster_subtype.groupby('PATID'):
        tmp = group[['new_PATID', col]].copy()
        filtered = tmp[tmp[col] == 1]
        if filtered.empty:
            continue
        try:
            first_timestep = int(filtered.iloc[0]['new_PATID'].split('_')[-1])
            records.append({'PATID': patid,
                            'first_comorbidity_days': first_timestep * 90})
        except (IndexError, ValueError):
            continue
    return pd.DataFrame(records)


def build_comorbidity_sv_data(df_demo, df_cluster_subtype,
                               comorbidity_cols=COMORBIDITY_COLS):
    """Survival data for logrank + Cox: only patients with comorbidity before AD.

    Follow-up start : first comorbidity occurrence
    Follow-up end   : AD diagnosis or 5 years (whichever comes first)
    event = 1 : AD diagnosis within 5 years of comorbidity onset
    event = 0 : right-censored at 5 years
    """
    patient_subtype = (df_cluster_subtype[['PATID', 'cls_pattern_encoded']]
                       .drop_duplicates(subset='PATID')
                       .rename(columns={'cls_pattern_encoded': 'subtype'}))

    demo_time = (df_demo[['ID', 'days_after_ENC']]
                 .rename(columns={'ID': 'PATID', 'days_after_ENC': 'days_to_AD'}))

    sv_dict = {}
    for col in tqdm(comorbidity_cols, desc="Building survival data (comorbidity subset)"):
        df_comorbidity = _extract_comorbidity_onset(df_cluster_subtype, col)

        merged = (patient_subtype
                  .merge(df_comorbidity, on='PATID', how='inner')
                  .merge(demo_time,      on='PATID', how='left'))

        merged['time'] = merged['days_to_AD'] - merged['first_comorbidity_days']
        merged = merged[merged['time'] > 0].reset_index(drop=True)

        merged['event'] = (merged['time'] <= OBSERVATION_WINDOW_DAYS).astype(int)
        merged['time']  = merged['time'].clip(upper=OBSERVATION_WINDOW_DAYS)

        # Binary sex group for Cox
        merged['sex_group'] = merged['subtype'].isin(MALE_DOMINANT_SUBTYPES).astype(int)

        n_total = len(merged)
        n_event = merged['event'].sum()
        print(f"  [{col}] n={n_total}, AD onset within 5 yrs={n_event} "
              f"({n_event/n_total:.1%}), censored={n_total-n_event} "
              f"({(n_total-n_event)/n_total:.1%})")

        sv_dict[col] = merged[['PATID', 'time', 'event', 'subtype', 'sex_group']]

    return sv_dict


def build_comorbidity_sv_data_all(df_demo, df_cluster_subtype,
                                   comorbidity_cols=COMORBIDITY_COLS):
    """Survival data for KM: ALL patients included.

    For patients WITH comorbidity before AD:
        time  = days from comorbidity onset to AD (clipped at 5 years)
        event = 1 if AD within 5 years, else 0
    For patients WITHOUT comorbidity before AD (no comorbidity, or comorbidity
    after AD):
        time  = OBSERVATION_WINDOW_DAYS (censored at 5-year boundary)
        event = 0

    This allows KM curves to include all 5 subtypes with all patients.
    Note: the 'time' variable has different biological origins across the two
    groups, so this dataset is intended for visualisation only.
    """
    patient_subtype = (df_cluster_subtype[['PATID', 'cls_pattern_encoded']]
                       .drop_duplicates(subset='PATID')
                       .rename(columns={'cls_pattern_encoded': 'subtype'}))

    demo_time = (df_demo[['ID', 'days_after_ENC']]
                 .rename(columns={'ID': 'PATID', 'days_after_ENC': 'days_to_AD'}))

    base = patient_subtype.merge(demo_time, on='PATID', how='left')

    sv_dict_all = {}
    for col in tqdm(comorbidity_cols, desc="Building survival data (all patients)"):
        df_comorbidity = _extract_comorbidity_onset(df_cluster_subtype, col)

        merged = base.merge(df_comorbidity, on='PATID', how='left')

        has_comorbidity_before_AD = (
            merged['first_comorbidity_days'].notna() &
            (merged['days_to_AD'] - merged['first_comorbidity_days'] > 0)
        )

        # Default: no comorbidity before AD → censored at observation window
        merged['time']  = OBSERVATION_WINDOW_DAYS
        merged['event'] = 0

        # Overwrite for patients with comorbidity before AD
        raw_time = (merged.loc[has_comorbidity_before_AD, 'days_to_AD']
                    - merged.loc[has_comorbidity_before_AD, 'first_comorbidity_days'])
        merged.loc[has_comorbidity_before_AD, 'event'] = (
            (raw_time <= OBSERVATION_WINDOW_DAYS).astype(int))
        merged.loc[has_comorbidity_before_AD, 'time'] = raw_time.clip(
            upper=OBSERVATION_WINDOW_DAYS)

        n_total  = len(merged)
        n_event  = merged['event'].sum()
        n_no_com = (~has_comorbidity_before_AD).sum()
        print(f"  [{col}] n_total={n_total}, with comorbidity before AD={has_comorbidity_before_AD.sum()}, "
              f"no comorbidity before AD={n_no_com} (censored at {OBSERVATION_WINDOW_DAYS}d)")

        sv_dict_all[col] = merged[['PATID', 'time', 'event', 'subtype']]

    return sv_dict_all


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def KM_analysis(sv_dict_all):
    """KM curves using all patients (comorbidity onset → AD, non-comorbidity
    patients censored at 5 years). 5 curves, one per subtype."""
    for col, sv_data in sv_dict_all.items():
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10, 6))
        for cluster_id, group in sv_data.groupby('subtype'):
            kmf.fit(durations=group['time'], event_observed=group['event'],
                    label=f'S{cluster_id}')
            kmf.plot_survival_function()
        plt.title(f'KM: Time from {col} Onset to AD Diagnosis\n'
                  f'(all patients; no-comorbidity censored at 5 yr)')
        plt.xlabel('Days from Comorbidity Onset (censored at 5 years)')
        plt.ylabel('Fraction without AD')
        plt.legend(title='Sub-phenotype')
        plt.grid()
        save_path = os.path.join(str(SAVE_DIR),
                                 f'step5_1_sv_{col.replace(" ", "_")}.pdf')
        plt.savefig(save_path)
        plt.show()


def logrank_analysis(sv_dict):
    """Multivariate log-rank test: 5 subtypes, comorbidity-subset patients."""
    results = {}
    for col, sv_data in sv_dict.items():
        result = multivariate_logrank_test(
            event_durations=sv_data['time'],
            groups=sv_data['subtype'],
            event_observed=sv_data['event'],
        )
        print(f"\n[{col}] Multivariate log-rank test (5 subtypes):")
        print(f"  n={len(sv_data)}, "
              f"test statistic={result.test_statistic:.4f}, "
              f"p-value={result.p_value:.4e}")
        result.print_summary()
        results[col] = result
    return results


def prepare_covariates(df_demo):
    """Demographic covariates keyed on patient ID.

    Excluded: AD_year (collinear with survival time).
    Excluded: SEX (subtypes are sex-specific; inclusion causes collinearity).
    """
    cov = df_demo[['ID', 'Age', 'race', 'HISPANIC', 'site', 'enc_count']].copy()
    cov['HISPANIC'] = (cov['HISPANIC'] == 'Y').astype(int)
    cov = pd.get_dummies(cov, columns=['race', 'site'], drop_first=True)
    return cov


def cox_analysis(sv_dict, df_demo):
    """Cox model per comorbidity.

    Main predictor : sex_group (1 = male-dominant S2/S3/S4,
                                ref = female-dominant S0/S1)
    Covariates     : Age, race, HISPANIC, site, enc_count
    """
    covariates = prepare_covariates(df_demo)
    results = {}
    for col, sv_data in sv_dict.items():
        merged = sv_data.merge(covariates, left_on='PATID', right_on='ID',
                               how='left')
        n_missing = merged[covariates.columns.drop('ID')].isna().any(axis=1).sum()
        if n_missing > 0:
            print(f"  [{col}] Warning: {n_missing} rows missing covariates — dropping.")
        merged = merged.dropna(subset=covariates.columns.drop('ID').tolist())

        cox_df = merged.drop(columns=['PATID', 'ID', 'subtype'])
        bool_cols = cox_df.select_dtypes(include='bool').columns
        cox_df[bool_cols] = cox_df[bool_cols].astype(int)

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_df, duration_col='time', event_col='event')
        print(f"\n[{col}] Cox PH model "
              f"(male-dominant vs female-dominant, "
              f"time from comorbidity to AD onset, right-censored at 5 years, adjusted):")
        cph.print_summary()

        print(f"\n[{col}] PH assumption check:")
        cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)

        results[col] = cph
    return results


# ---------------------------------------------------------------------------
# Landmark Analysis
# ---------------------------------------------------------------------------

def build_landmark_sv_data(df_demo, df_cluster_subtype,
                            comorbidity_cols=COMORBIDITY_COLS,
                            landmark_days=LANDMARK_DAYS,
                            window_days=LANDMARK_WINDOW_DAYS):
    """Landmark survival data.

    Time origin : first EHR encounter
    Landmark L  : landmark_days (default 1 year) after first EHR encounter
    Restriction : only patients who have NOT yet developed AD by time L
                  (days_to_AD > landmark_days) — eliminates immortal time bias
    Follow-up   : from L to AD onset, right-censored at L + window_days
    event = 1   : AD onset within window after landmark
    event = 0   : right-censored at window_days after landmark

    For each comorbidity, patients are also labelled by whether they had
    developed that comorbidity by the landmark time (comorbidity_at_landmark).
    """
    patient_subtype = (df_cluster_subtype[['PATID', 'cls_pattern_encoded']]
                       .drop_duplicates(subset='PATID')
                       .rename(columns={'cls_pattern_encoded': 'subtype'}))

    demo_time = (df_demo[['ID', 'days_after_ENC']]
                 .rename(columns={'ID': 'PATID', 'days_after_ENC': 'days_to_AD'}))

    base = patient_subtype.merge(demo_time, on='PATID', how='left')

    # Restrict to patients who have not yet developed AD by landmark time
    base = base[base['days_to_AD'] > landmark_days].reset_index(drop=True)

    # Time from landmark to AD onset; clip at follow-up window
    raw_time = base['days_to_AD'] - landmark_days
    base['time']  = raw_time.clip(upper=window_days)
    base['event'] = (raw_time <= window_days).astype(int)

    sv_dict_landmark = {}
    for col in tqdm(comorbidity_cols, desc="Building landmark survival data"):
        df_comorbidity = _extract_comorbidity_onset(df_cluster_subtype, col)

        merged = base.merge(df_comorbidity, on='PATID', how='left')

        # Comorbidity status AT landmark: had comorbidity by time L?
        merged['comorbidity_at_landmark'] = (
            merged['first_comorbidity_days'].notna() &
            (merged['first_comorbidity_days'] <= landmark_days)
        ).astype(int)

        n_total   = len(merged)
        n_with    = merged['comorbidity_at_landmark'].sum()
        n_event   = merged['event'].sum()
        print(f"  [{col}] landmark={landmark_days}d, "
              f"AD-free at landmark={n_total}, "
              f"comorbidity by landmark={n_with} ({n_with/n_total:.1%}), "
              f"AD within {window_days}d after landmark={n_event} ({n_event/n_total:.1%})")

        sv_dict_landmark[col] = merged[['PATID', 'time', 'event',
                                        'subtype', 'comorbidity_at_landmark']]

    return sv_dict_landmark


def landmark_KM_analysis(sv_dict_landmark, landmark_days=LANDMARK_DAYS):
    """Two-panel KM plot from landmark time for each comorbidity.

    Left panel  : KM by subtype (5 curves, all patients at landmark)
    Right panel : KM by comorbidity status at landmark
                  (with vs. without comorbidity by time L)
    """
    for col, sv_data in sv_dict_landmark.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ── Panel 1: by subtype ──────────────────────────────────────────────
        kmf = KaplanMeierFitter()
        ax = axes[0]
        for cluster_id, group in sv_data.groupby('subtype'):
            kmf.fit(durations=group['time'], event_observed=group['event'],
                    label=f'S{cluster_id}')
            kmf.plot_survival_function(ax=ax)
        ax.set_title(f'Landmark KM by Sub-phenotype\n'
                     f'[{col}]\n'
                     f'(landmark = {landmark_days // 365} yr after first EHR)')
        ax.set_xlabel(f'Days from Landmark (right-censored at {LANDMARK_WINDOW_DAYS // 365} yr)')
        ax.set_ylabel('Fraction without AD')
        ax.legend(title='Sub-phenotype')
        ax.grid()

        # ── Panel 2: by comorbidity status at landmark ───────────────────────
        kmf2 = KaplanMeierFitter()
        ax2 = axes[1]
        labels = {0: 'No comorbidity by landmark',
                  1: f'Has {col} by landmark'}
        for com_status, group in sv_data.groupby('comorbidity_at_landmark'):
            kmf2.fit(durations=group['time'], event_observed=group['event'],
                     label=labels[com_status])
            kmf2.plot_survival_function(ax=ax2)
        ax2.set_title(f'Landmark KM by Comorbidity Status\n'
                      f'[{col}]\n'
                      f'(landmark = {landmark_days // 365} yr after first EHR)')
        ax2.set_xlabel(f'Days from Landmark (right-censored at {LANDMARK_WINDOW_DAYS // 365} yr)')
        ax2.set_ylabel('Fraction without AD')
        ax2.legend(title='Status at landmark')
        ax2.grid()

        plt.tight_layout()
        save_path = os.path.join(
            str(SAVE_DIR),
            f'step5_1_landmark_{col.replace(" ", "_")}_L{landmark_days}d.pdf')
        plt.savefig(save_path)
        plt.show()
        print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df_demo = pd.read_csv(STEP0_DEMO_CSV)
    df_cluster_subtype = pd.read_csv(STEP4_CLUSTER_SUBTYPE_CSV)

    race_dict = {'02': 'Asian', '03': 'Black', '05': 'White'}
    df_demo['race'] = df_demo['RACE'].map(race_dict).fillna('Other')
    df_demo['site'] = df_demo['SOURCE_masked']

    # KM: all patients (non-comorbidity patients censored at 5 yr boundary)
    sv_dict_all = build_comorbidity_sv_data_all(df_demo, df_cluster_subtype,
                                                 COMORBIDITY_COLS)
    KM_analysis(sv_dict_all)

    # Log-rank + Cox: comorbidity-subset patients only
    sv_dict = build_comorbidity_sv_data(df_demo, df_cluster_subtype,
                                         COMORBIDITY_COLS)
    logrank_results = logrank_analysis(sv_dict)
    cox_results     = cox_analysis(sv_dict, df_demo)

    # Landmark Analysis: restrict to patients AD-free at landmark (1 yr),
    # then track from landmark to AD onset (right-censored at 5 yr)
    print("\n" + "=" * 60)
    print("Landmark Analysis (landmark = 1 yr from first EHR)")
    print("=" * 60)
    sv_dict_landmark = build_landmark_sv_data(df_demo, df_cluster_subtype,
                                               COMORBIDITY_COLS,
                                               landmark_days=LANDMARK_DAYS,
                                               window_days=LANDMARK_WINDOW_DAYS)
    landmark_KM_analysis(sv_dict_landmark, landmark_days=LANDMARK_DAYS)


if __name__ == '__main__':
    main()
