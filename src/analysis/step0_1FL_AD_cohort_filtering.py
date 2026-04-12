# -------------- 1. load packages -------------- #
import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import RAW_DATA_DIR, STEP0_DEMO_CSV, ensure_dir

# Antidementia medications: Donepezil, Galantamine, Rivastigmine, Memantine, Donepezil+Memantine, Aducanumab, Lecanemab, Donanemab, Brexpiprazole, Benzgalantamine
rxcuis = [
    "2626143", "2626142", "2626149", "2626144", "2723403", "2626150",
    "2723408", "2723406", "2626147", "2626155", "2626153", "2626156",
    "2723411", "2626145", "2659968", "2626151",
    "2687966", "2687965", "2687974", "2687967", "2687975", "2687972",
    "2687979", "2687968", "2687970", "2687976",
    "135447", "236559", "483068", "997220", "997226", "483073",
    "371957", "997223", "310009", "1100184", "997229", "310010",
    "1599802", "1599805", "1805420", "1599803", "1805425", "1858971",
    "997222", "997228", "1602588", "1100187", "1602594", "997224",
    "997230",
    "183379", "373797", "314215", "312836", "994808", "314214",
    "312835", "374628", "312834", "1805978", "725021", "725023",
    "1308569", "226667", "1308571", "226665", "226668", "226666",
    "751302", "725105",
    "4637", "860693", "583097", "860695", "860707", "860715",
    "583132", "384642", "860901", "384641", "579148", "310436",
    "310437", "602734", "860697", "860709", "602737", "860717",
    "602736",
    "6719", "996572", "754513", "996624", "996593", "996594",
    "996603", "996609", "996615", "577156", "996740", "372757",
    "996561", "250507", "996571", "403932",
    "1599802", "1599805", "1805420", "1599803", "1805425",
    "236685", "1858971", "1602588", "996633", "996634", "996597",
    "1602594", "996605", "996742", "996611", "996574", "996563",
    "996617"
]

# -------------- 2. set input and output paths -------------- #

input_folder = str(RAW_DATA_DIR) + '/'
output_folder = str(ensure_dir(STEP0_DEMO_CSV.parent)) + '/'


# -------------- 3. main function -------------- #
def main():
    # 1. load all patients' demographics data
    df_demo = pd.read_csv(open(input_folder + 'DEMOGRAPHIC.csv'), skiprows=[1], usecols=['ID', 'BIRTH_DATE', 'SEX', 'HISPANIC', 'RACE'], dtype=str)
    print('# Total Patients:', df_demo['ID'].unique().shape[0])

    # 2. Check if patients have AD diagnosis through diagnosis file
    # # 2.1 load diagnosis and prescription file
    df_diag = pd.read_csv(open(input_folder+'DIAGNOSIS.csv'), skiprows=[1], usecols=['ID', 'ADMIT_DATE', 'DX', 'DX_TYPE'], dtype=str) # 5-7 mins to load this file
    print("# the number of diagnosis records in diagnosis file", df_diag.shape[0])
    print("# Patients in diagnosis file:", df_diag['ID'].unique().shape[0])

    df_pres = pd.read_csv(open(input_folder+"PRESCRIBING.csv"), skiprows=[1], usecols=['ID', 'RX_ORDER_DATE', 'RX_START_DATE', 'RXNORM_CUI', 'RX_SOURCE'], dtype=str)
    print("# the number of prescription records in prescription file", df_pres.shape[0])
    print("# Patients in prescription file:", df_pres['ID'].unique().shape[0])

    # # 2.2 determine first AD onset date
    # Alzheimer's Disease: ICD9: 331.0; ICD10: F00, G30.0, G30.1, G30.8, G30.9
    tmp_AD = df_diag[df_diag['DX'].isin(['3310', '331.0', 'G30.0', 'G30.1', 'G30.8', 'G30.9']) | df_diag['DX'].str.startswith(('G30')) | df_diag['DX'].str.startswith(('F00'))]
    tmp_AD['ADMIT_DATE'] = pd.to_datetime(tmp_AD['ADMIT_DATE'])
    AD_min_date = tmp_AD.groupby('ID')['ADMIT_DATE'].agg(min)
    AD_max_date = tmp_AD.groupby('ID')['ADMIT_DATE'].agg(max)
    AD_min_to_max_days = (AD_max_date - AD_min_date).dt.days

    # # 2.3 merge first and last AD onset date to demographics data
    df_demo = pd.merge(df_demo, AD_min_date, on='ID', how='left')
    df_demo = df_demo.rename(columns={'ADMIT_DATE': 'first_AD_date'})
    df_demo = pd.merge(df_demo, AD_max_date, on='ID', how='left')
    df_demo = df_demo.rename(columns={'ADMIT_DATE': 'last_AD_date'})
    df_demo = pd.merge(df_demo, AD_min_to_max_days, on='ID', how='left')
    df_demo = df_demo.rename(columns={'ADMIT_DATE': 'AD_first_to_last_days'})
    
    df_pres.dropna(subset=['RXNORM_CUI'], inplace=True)
    df_pres['RXNORM_CUI'] = df_pres['RXNORM_CUI'].astype(str)
    df_pres = df_pres[df_pres['RXNORM_CUI'].isin(rxcuis)]
    df_pres['RX_ORDER_DATE'] = pd.to_datetime(df_pres['RX_ORDER_DATE'], errors='coerce')
    df_pres_ad = df_pres[df_pres['ID'].isin(df_demo['ID'])]
    df_pres_merged = df_demo[['ID', 'first_AD_date']].merge(
        df_pres_ad[['ID', 'RX_ORDER_DATE', 'RXNORM_CUI']], 
        on='ID', 
        how='left'
    )
    
    df_pres_merged['AD_med_interval'] = (df_pres_merged['RX_ORDER_DATE'] - df_pres_merged['first_AD_date']).dt.days
    df_pres_merged = df_pres_merged[(df_pres_merged['AD_med_interval'] >= 0) & (df_pres_merged['AD_med_interval'] <= 180)]
    ad_med_dates = df_pres_merged.groupby('ID')['RX_ORDER_DATE'].min().reset_index()
    ad_med_dates.columns = ['ID', 'AD_med_date']
    df_demo = df_demo.merge(ad_med_dates, on='ID', how='left')
    df_demo = df_demo[(df_demo['AD_med_date'].notna())]

    print('# Patients have AD diagnosis:', df_demo['ID'].unique().shape[0])

    # Clean up temporary data in memory to prevent kernel crashes
    del df_diag
    del tmp_AD
    del AD_min_date
    del AD_max_date
    del AD_min_to_max_days
    del df_pres_ad
    del df_pres
    del df_pres_merged
    del ad_med_dates

    # 3. check death data and encounter data
    # 3.1 load death data
    df_death = pd.read_csv(open(input_folder+'DEATH.csv'), skiprows=[1], usecols=['ID', 'DEATH_DATE'], dtype=str)
    # By default, keep the first one.
    df_death = df_death.sort_values('DEATH_DATE').drop_duplicates(subset=['ID'], keep='first')
    df_demo = pd.merge(df_demo, df_death, on='ID', how='left')
    # 3.2 load encounter data
    df_enc = pd.read_csv(input_folder + 'ENCOUNTER.csv', skiprows=[1], usecols=['ID', 'ADMIT_DATE'], low_memory=False) # 5-7 mins to load this file
    print("-------------------- loaded ENCOUNTER.csv --------------------") # 12 mins to process
    enc_min_date = df_enc.groupby('ID')['ADMIT_DATE'].agg(min)
    enc_max_date = df_enc.groupby('ID')['ADMIT_DATE'].agg(max)

    df_demo = pd.merge(df_demo, enc_min_date, on='ID', how='inner')
    df_demo = df_demo.rename(columns={'ADMIT_DATE': 'first_ENC_date'})
    df_demo = pd.merge(df_demo, enc_max_date, on='ID', how='inner')
    df_demo = df_demo.rename(columns={'ADMIT_DATE': 'last_ENC_date'})

    df_demo['days_before_first_AD'] = (pd.to_datetime(df_demo['first_AD_date']) - pd.to_datetime(df_demo['first_ENC_date'])).dt.days
    df_demo['days_after_first_AD'] = (pd.to_datetime(df_demo['last_ENC_date']) - pd.to_datetime(df_demo['first_AD_date'])).dt.days
    df_demo['days_all_followup'] = (pd.to_datetime(df_demo['last_ENC_date']) - pd.to_datetime(df_demo['first_ENC_date'])).dt.days

    del df_enc
    del df_death
    del enc_min_date
    del enc_max_date

    # 4. filter patients by criteria:
    # 4.1. age at first encounter >= 50
    # 4.2. have no death records
    # 4.3. first encounter date is not before 2012-01-01
    # 4.4. last encounter date is not after 2024-07-01
    df_demo['BIRTH_DATE'] = pd.to_datetime(df_demo['BIRTH_DATE'])
    df_demo['first_AD_date'] = pd.to_datetime(df_demo['first_AD_date'])
    df_demo['last_AD_date'] = pd.to_datetime(df_demo['last_AD_date'])
    df_demo['DEATH_DATE'] = pd.to_datetime(df_demo['DEATH_DATE'])
    df_demo['first_ENC_date'] = pd.to_datetime(df_demo['first_ENC_date'])
    df_demo['last_ENC_date'] = pd.to_datetime(df_demo['last_ENC_date'])
    df_demo['Age'] = (df_demo['first_ENC_date'] - df_demo['BIRTH_DATE']).dt.days // 365
    df_demo = df_demo[df_demo['Age'] >= 50]
    df_demo = df_demo[pd.to_datetime(df_demo['first_ENC_date']) >= pd.Timestamp('2012-01-01')]
    df_demo = df_demo[pd.to_datetime(df_demo['last_ENC_date']) <= pd.Timestamp('2024-07-01')]
    # 4.5. follow-up days before first AD date is 3 years or more
    df_demo = df_demo[df_demo['days_before_first_AD'] >= 1095]
    # 4.6. follow-up days after first AD date is 1 year or more
    df_demo = df_demo[df_demo['days_after_first_AD'] >= 365]

    # 5. save the filtered cohort
    print('# Filtered Patients:', df_demo['ID'].unique().shape[0])
    df_demo.to_csv(STEP0_DEMO_CSV, index=False)

if __name__ == '__main__':
    main()
