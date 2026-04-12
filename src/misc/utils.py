from __future__ import annotations

import pandas as pd

from project_paths import mapping_file


def _load_icd_to_phecode(df_diag: pd.DataFrame) -> pd.DataFrame:
    df_diag9 = df_diag[df_diag["DX_TYPE"].isin(["09", 9])].copy()
    df_diag10 = df_diag[df_diag["DX_TYPE"].isin(["10", 10])].copy()

    df_map9 = pd.read_csv(mapping_file("phecode_icd9_map_unrolled.csv"))
    df_diag9 = pd.merge(df_diag9, df_map9, left_on="DX", right_on="icd9", how="inner")

    df_map10 = pd.read_csv(mapping_file("Phecode_map_v1_2_icd10_beta.csv"))
    df_diag10["DX"] = df_diag10["DX"].astype(str).str[:5]
    df_diag10 = pd.merge(df_diag10, df_map10[["ICD10", "PHECODE"]], left_on="DX", right_on="ICD10", how="left")
    df_diag10 = df_diag10.rename(columns={"PHECODE": "phecode"})
    df_map10 = df_map10.rename(columns={"PHECODE": "phecode"})

    df_diag10_na = df_diag10[df_diag10["ICD10"].isna()].copy()
    df_diag10_na["DX"] = df_diag10_na["DX"].astype(str).str[:3]
    df_diag10_na = df_diag10_na.drop(columns=["ICD10", "phecode"])
    df_diag10_na = pd.merge(df_diag10_na, df_map10[["ICD10", "phecode"]], left_on="DX", right_on="ICD10", how="left")

    df_diag10 = pd.concat([df_diag10, df_diag10_na], ignore_index=True)
    df_diag10 = df_diag10[~df_diag10["phecode"].isna()]

    df_diag_phecode = pd.concat([df_diag9[["PATID", "phecode"]], df_diag10[["PATID", "phecode"]]], ignore_index=True)
    df_diag_phecode = df_diag_phecode.drop_duplicates(keep="first")
    df_diag_phecode = df_diag_phecode[~df_diag_phecode["phecode"].isna()]
    return df_diag_phecode


def _load_phecode_names(df_diag_phecode: pd.DataFrame) -> pd.DataFrame:
    df_phenotype = pd.read_csv(mapping_file("phecode_definitions1.2.csv"))
    df_diag_phecode = pd.merge(df_diag_phecode, df_phenotype[["phecode", "phenotype"]], on="phecode", how="inner")
    df_diag_phecode = df_diag_phecode[["PATID", "phenotype"]].drop_duplicates(keep="first")
    return df_diag_phecode[~df_diag_phecode["phenotype"].isna()]


def encode_diag_to_phecode(output_folder, df_demo, df_diag, unique_phecode_list):
    del output_folder
    df_diag_phecode = _load_phecode_names(_load_icd_to_phecode(df_diag))
    df_diag_code = df_demo[["PATID"]].copy()
    for code in unique_phecode_list:
        tmp = df_diag_phecode[df_diag_phecode["phenotype"] == code]
        df_diag_code[f"Phe_{code}"] = df_diag_code["PATID"].isin(tmp["PATID"]).astype(int)
    return df_diag_code


def get_diag_all_unique_phecode(output_folder, df_demo, df_diag):
    del output_folder, df_demo
    df_diag_phecode = _load_phecode_names(_load_icd_to_phecode(df_diag))
    return df_diag_phecode["phenotype"].unique()


def _load_atc_drug_table(df_demo, df_disp, df_pres):
    df_pres = pd.merge(df_demo[["PATID", "CUT_date"]], df_pres, on="PATID", how="inner")
    df_pres = df_pres[["PATID", "RXNORM_CUI"]].drop_duplicates(keep="first").dropna(axis=0, how="any")

    df_disp = pd.merge(df_demo[["PATID", "CUT_date"]], df_disp, on="PATID", how="inner")
    df_disp = df_disp[["PATID", "NDC"]].drop_duplicates(keep="first").dropna(axis=0, how="any")

    map_ndc_atc = pd.read_csv(mapping_file("NDC_to_ATC.text"), sep="|", header=None, dtype=str)
    map_ndc_atc = map_ndc_atc[[3, 19]]
    map_ndc_atc.columns = ["NDC", "ATC"]
    map_ndc_atc = map_ndc_atc[~map_ndc_atc["ATC"].isna()]

    map_rxnorm_atc = pd.read_csv(mapping_file("RXNORM_to_ATC.text"), sep="|", header=None, dtype=str)
    map_rxnorm_atc = map_rxnorm_atc[[3, 15]]
    map_rxnorm_atc.columns = ["RXNORM_CUI", "ATC"]
    map_rxnorm_atc = map_rxnorm_atc[~map_rxnorm_atc["ATC"].isna()]

    df_pres = pd.merge(df_pres, map_rxnorm_atc, on="RXNORM_CUI", how="inner")
    df_disp = pd.merge(df_disp, map_ndc_atc, on="NDC", how="inner")

    df_drug = pd.concat([df_pres[["PATID", "ATC"]], df_disp[["PATID", "ATC"]]], ignore_index=True)
    df_drug["ATC"] = df_drug["ATC"].astype(str).str[:4]

    df_atc_name = pd.read_csv(mapping_file("ATC3rd_name.csv"), usecols=["ATC_name", "ATC"])
    df_drug = pd.merge(df_drug, df_atc_name, on="ATC", how="inner")
    return df_drug[["PATID", "ATC_name"]].drop_duplicates(keep="first")


def encode_drug_to_ATC(output_folder, df_demo, df_disp, df_pres, unique_ATC_list):
    del output_folder
    df_drug = _load_atc_drug_table(df_demo, df_disp, df_pres)
    df_drug_code = df_demo[["PATID"]].copy()
    for code in unique_ATC_list:
        tmp = df_drug[df_drug["ATC_name"] == code]
        df_drug_code[f"ATC_{code}"] = df_drug_code["PATID"].isin(tmp["PATID"]).astype(int)
    return df_drug_code


def get_drug_all_unique_ATC(output_folder, df_demo, df_disp, df_pres):
    del output_folder
    df_drug = _load_atc_drug_table(df_demo, df_disp, df_pres)
    return df_drug["ATC_name"].unique()
