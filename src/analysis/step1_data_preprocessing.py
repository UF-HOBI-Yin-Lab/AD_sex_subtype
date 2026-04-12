# -------------- 1. load packages -------------- #
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from tqdm import tqdm
import numpy as np
import json
import re
import pickle
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.utils import encode_diag_to_phecode, encode_drug_to_ATC, get_drug_all_unique_ATC, get_diag_all_unique_phecode
from project_paths import RAW_DATA_DIR, STEP0_DEMO_CSV, STEP1_DIR, STEP1_PATIENT_PKL, STEP1_SUBSEQ_NPZ, STEP1_3D_NPZ, ensure_dir

warnings.filterwarnings('ignore')

# -------------- 2. set input and output paths -------------- #
input_folder = str(STEP0_DEMO_CSV.parent.parent) + '/'
output_folder = str(ensure_dir(STEP1_DIR)) + '/'

# extract unique codes for diagnosis, dispensing and prescription for mapping in data extraction
def extract_unique_codes(df_demo, df_diag, df_disp, df_pres, max_seq_len, interval=90):
    # Get included phecodes list
    df_demo['CUT_date'] = pd.to_datetime(df_demo['first_ENC_date']) + timedelta(days=max_seq_len*interval)

    # Get included phecode list
    df_diag1 = pd.merge(df_demo, df_diag, on='PATID', how='inner')
    df_diag1 = df_diag1[(df_diag1['ADMIT_DATE'] <= df_diag1['CUT_date'])]
    df_diag1.drop_duplicates(keep='first', inplace=True)
    unique_phecode_list = get_diag_all_unique_phecode(output_folder, df_demo, df_diag1)

    # Get included ATC list
    df_disp1 = pd.merge(df_demo[['PATID', 'CUT_date']], df_disp, on='PATID', how='inner')
    df_disp1 = df_disp1[(df_disp1['DISPENSE_DATE'] <= df_disp1['CUT_date'])]
    df_disp1 = df_disp1[['PATID', 'NDC']]
    df_disp1.drop_duplicates(keep='first', inplace=True)
    df_disp1 = df_disp1.dropna(axis=0, how='any')

    df_pres1 = pd.merge(df_demo[['PATID', 'CUT_date']], df_pres, on='PATID', how='inner')
    df_pres1 = df_pres1[(df_pres1['RX_START_DATE'] <= df_pres1['CUT_date'])]
    df_pres1 = df_pres1[['PATID', 'RXNORM_CUI']]
    df_pres1.drop_duplicates(keep='first', inplace=True)
    df_pres1 = df_pres1.dropna(axis=0, how='any')

    unique_ATC_list = get_drug_all_unique_ATC(output_folder, df_demo, df_disp1, df_pres1)

    return unique_phecode_list, unique_ATC_list

# extract data for each time interval
def extract_data(df_demo, df_demo_code, df_diag, df_disp, df_pres, unique_phecode_list, unique_ATC_list, max_seq_len, interval=90):
    data_dict = {}
    for idx in tqdm(range(max_seq_len), desc='Extracting data'):
        # print(idx)
        
        df_demo['CUT_date'] = pd.to_datetime(df_demo['first_ENC_date']) + timedelta(days=idx*interval)
        
        # Diagnosis code
        df_diag1 = pd.merge(df_demo, df_diag, on='PATID', how='inner')
        df_diag1 = df_diag1[(df_diag1['ADMIT_DATE'] <= df_diag1['CUT_date'])]
        df_diag1.drop_duplicates(keep='first', inplace=True)
        df_diag1 = df_diag1.dropna(axis=0, how='any')
        
        # Diagnosis code mapping
        df_diag_code = encode_diag_to_phecode(output_folder, df_demo, df_diag1, unique_phecode_list)
        
        # Drug code 1 NDC
        df_disp1 = pd.merge(df_demo[['PATID', 'CUT_date']], df_disp, on='PATID', how='inner')
        df_disp1 = df_disp1[(df_disp1['DISPENSE_DATE'] <= df_disp1['CUT_date'])]
        df_disp1 = df_disp1[['PATID', 'NDC']]
        df_disp1.drop_duplicates(keep='first', inplace=True)
        df_disp1 = df_disp1.dropna(axis=0, how='any')
        
        # Drug code 2 RxNorm
        df_pres1 = pd.merge(df_demo[['PATID', 'CUT_date']], df_pres, on='PATID', how='inner')
        df_pres1 = df_pres1[(df_pres1['RX_START_DATE'] <= df_pres1['CUT_date'])]
        df_pres1 = df_pres1[['PATID', 'RXNORM_CUI']]
        df_pres1.drop_duplicates(keep='first', inplace=True)
        df_pres1 = df_pres1.dropna(axis=0, how='any')
        
        # Drug code mapping 
        df_drug_code = encode_drug_to_ATC(output_folder, df_demo, df_disp1, df_pres1, unique_ATC_list)
        
        df_code_all = pd.concat([df_diag_code[['PATID']], df_demo_code, df_diag_code.iloc[:,1:], df_drug_code.iloc[:,1:]], axis=1)
        data_dict[idx] = df_code_all
    
    return data_dict, df_code_all

# extract data for each patient
def extract_patient_data(df_demo, data_dict, max_seq_len, interval=90):
    data_dict_by_ID = {}
    for ID in tqdm(df_demo.PATID, desc='Extracting patient-wise sequences'):
        seq_len = int(df_demo[df_demo['PATID']==ID]['days_ENC'].values[0]/interval)
        for idx in range(max_seq_len):
            if idx < seq_len:
                tmp = data_dict[idx]
                if idx==0:
                    row = tmp[tmp['PATID']==ID].iloc[:, 1:].values 
                else:
                    row = np.vstack([row, tmp[tmp['PATID']==ID].iloc[:, 1:].values])
        data_dict_by_ID[ID] = row
    return data_dict_by_ID

# extract subsequence data for training
def extract_subsequence_data(df_demo, data_dict_by_patient, max_seq_len, subsequence_length=2):
    """
    Extract subsequence data and convert to 3D array directly to save memory.
    Returns both dict format (for indexing) and 3D array (for training).
    """
    # Step 1: Calculate total number of subsequences
    total_subseqs = 0
    for ID in df_demo.PATID:
        seq_len = data_dict_by_patient[ID].shape[0]
        total_subseqs += (seq_len + subsequence_length - 1) // subsequence_length
    
    # Step 2: Get feature dimension
    first_id = df_demo.PATID.iloc[0]
    n_features = data_dict_by_patient[first_id].shape[1]
    
    # Step 3: Pre-allocate 3D array
    data_x = np.zeros((total_subseqs, max_seq_len, n_features), dtype=np.float32)
    data_y_list = []
    patid_list = []
    
    # Step 4: Fill the 3D array
    global_idx = 0
    data_sub_dict_by_ID = {}  # Keep dict for indexing (will be deleted after save)
    
    for ID in tqdm(df_demo.PATID, desc='Extracting subsequence data'):
        seq_len = data_dict_by_patient[ID].shape[0]
        label = df_demo[df_demo['PATID']==ID]['label'].values[0]
        
        local_idx = 0
        for i in range(1, seq_len+1, subsequence_length):
            # Extract subsequence
            tmp_array = data_dict_by_patient[ID][:i,:]
            
            # Fill into 3D array (padding with zeros automatically done by np.zeros)
            data_x[global_idx, :tmp_array.shape[0], :] = tmp_array
            
            # Store label and patid
            subseq_key = f'{ID}_{local_idx}'
            data_y_list.append(label)
            patid_list.append(subseq_key)
            
            # Store in dict (only for saving with index, will be deleted)
            data_sub_dict_by_ID[subseq_key] = data_x[global_idx]
            
            global_idx += 1
            local_idx += 1
    
    print('# Subsequences:',len(data_sub_dict_by_ID))
    
    # Step 5: Create label DataFrame
    data_label = pd.DataFrame({'label': data_y_list}, index=patid_list)
    
    return data_sub_dict_by_ID, data_label, data_x

def main():
    # 1. load demographics data from step0_1FL_AD_cohort_filtering.py
    df_demo = pd.read_csv(STEP0_DEMO_CSV)
    print(df_demo)

    df_demo['label'] = (df_demo['SEX'] == 'F').astype(int)

    # 2. extract data in 3 months intervals
    # 2.1 Check max sequence lenght, time intervel is set to 3 months
    # print('# Max days of data:', df_demo['days_all_followup'].max()) # CHANGE HERE
    # max_seq_len = int(df_demo['days_all_followup'].max()/90)
    print('# Max days of data:', df_demo['days_ENC'].max())
    max_seq_len = int(df_demo['days_ENC'].max()/90)
    print('# Max sequence length:', max_seq_len)

    # 2.2 Read data and find unique diagnosis and medication codes
    df_diag = pd.read_csv(RAW_DATA_DIR / 'DIAGNOSIS.csv', usecols=['ID', 'ADMIT_DATE', 'DX', 'DX_TYPE'], low_memory=False, dtype=str)
    df_disp = pd.read_csv(RAW_DATA_DIR / 'DISPENSING.csv', usecols=['ID', 'NDC', 'DISPENSE_DATE'], low_memory=False, dtype=str)
    df_pres = pd.read_csv(RAW_DATA_DIR / 'PRESCRIBING.csv', usecols=['ID', 'RXNORM_CUI', 'RX_START_DATE'], low_memory=False, dtype=str)

    df_diag.rename(columns={'ID': 'PATID'}, inplace=True)
    df_disp.rename(columns={'ID': 'PATID'}, inplace=True)
    df_pres.rename(columns={'ID': 'PATID'}, inplace=True)
    df_demo.rename(columns={'ID': 'PATID'}, inplace=True)
    # 2.3 Extract unique phecode and ATC codes
    df_diag = pd.merge(df_demo[['PATID', 'first_AD_date', 'first_ENC_date', 'last_ENC_date']], df_diag)
    df_diag = df_diag[~df_diag['first_AD_date'].isna()]
    df_diag = df_diag[(df_diag['ADMIT_DATE'] > df_diag['first_ENC_date']) & (df_diag['ADMIT_DATE'] < df_diag['last_ENC_date'])]

    df_disp = pd.merge(df_demo[['PATID', 'first_AD_date', 'first_ENC_date', 'last_ENC_date']], df_disp)
    df_disp = df_disp[~df_disp['first_AD_date'].isna()]
    df_disp = df_disp[(df_disp['DISPENSE_DATE'] > df_disp['first_ENC_date']) & (df_disp['DISPENSE_DATE'] < df_disp['last_ENC_date'])]

    df_pres = pd.merge(df_demo[['PATID', 'first_AD_date', 'first_ENC_date', 'last_ENC_date']], df_pres)
    df_pres = df_pres[~df_pres['first_AD_date'].isna()]
    df_pres = df_pres[(df_pres['RX_START_DATE'] > df_pres['first_ENC_date']) & (df_pres['RX_START_DATE'] < df_pres['last_ENC_date'])]

    unique_phecode_list, unique_ATC_list = extract_unique_codes(df_demo, df_diag, df_disp, df_pres, max_seq_len)
    
    # 3. demographics data preprocessing
    # 3.1 Age groups
    df_demo['age_groups'] = pd.cut(df_demo['Age'], bins=[49, 59, 69, 79, np.inf])
    df_age = pd.get_dummies(df_demo['age_groups'])
    print(df_demo.groupby('age_groups')['age_groups'].agg('count'), '\n')

    # 3.2 Sex (sex is label which is not included in the data)

    # 3.3 Race
    df_RACE = pd.get_dummies(df_demo['RACE'])
    print(df_demo.groupby('RACE')['RACE'].agg('count'), '\n')
    print('01=American Indian or Alaska Native, 02=Asian, 03=Black or African American, 04=Native, Hawaiian or Other Pacific Islander, ' 
        '05=White, 06=Multiple race, 07=Refuse to answer, NI=No information, UN=Unknown, OT=Other', '\n')

    print('Combine the unknown race groups (07, NI, UN) to Unknown')
    #df_demo.loc[df_demo['RACE'].isin(['07', 'NI', 'UN']), ['RACE']]='Unknown_race'
    df_RACE = pd.get_dummies(df_demo['RACE'])
    print('New race groups:', df_demo.groupby('RACE')['RACE'].agg('count'))

    df_demo_code = pd.concat([df_age, df_RACE], axis=1)

    # 4. extract sequence data in 3 months intervals
    # 4.1 Extract data to dataframe for each 3-month time interval with time interval as Keys
    data_dict_by_time, df_code_all = extract_data(df_demo, df_demo_code, df_diag, df_disp, df_pres, unique_phecode_list, unique_ATC_list, max_seq_len)
    print(f'the number of sequences: {len(data_dict_by_time)}')
    print(f'the dimension of each sequence: {data_dict_by_time[0].shape}')
    print(data_dict_by_time[0].head())

    # 4.2 reshape data to patient-wise sequences with PATID as Keys
    data_dict_by_patient = extract_patient_data(df_demo, data_dict_by_time, max_seq_len)
    print('# the number of patient-wise sequences:',len(data_dict_by_patient))
    first_key = list(data_dict_by_patient.keys())[0]
    print(f'the dimension of each sequence: {data_dict_by_patient[first_key].shape}')
    print(data_dict_by_patient[first_key][:5])
    del data_dict_by_time

    # 4.3 save data_dict_by_patient to pickle file
    with open(STEP1_PATIENT_PKL, 'wb') as f:
        pickle.dump(data_dict_by_patient, f)

    # 5. extract subsequence data, label and training/validation/testing set
    # 5.1 Extract subsequence data (returns dict for indexing + 3D array directly)
    data_sub_dict_by_ID, data_label, data_x = extract_subsequence_data(df_demo, data_dict_by_patient, max_seq_len)
    
    # Release memory: delete data_dict_by_patient (18GB saved!)
    del data_dict_by_patient
    import gc
    gc.collect()
    
    print('# the number of subsequences:', len(data_sub_dict_by_ID))
    print(f'3D array shape: {data_x.shape}')
    print(data_label.head())
    
    # Save dict format with index (for reference, use compressed to save disk space)
    np.savez_compressed(STEP1_SUBSEQ_NPZ, 
                       data_x=data_sub_dict_by_ID, data_y=data_label, 
                       PATID=data_label.index, col_name=df_code_all.columns)
    
    # Release memory: delete dict (only keep 3D array for training)
    del data_sub_dict_by_ID
    gc.collect()

    # 5.2 Create 3D label array (replicate label across all timesteps)
    # Shape: (n_samples, max_seq_len, 1)
    data_y = np.tile(data_label['label'].values.reshape(-1, 1, 1), (1, max_seq_len, 1)).astype(np.float32)
    print('Matrix size:', data_x.shape, data_y.shape)

    # Save 3D array (use compressed format to save disk space)
    y_type = np.array('categorical', dtype='<U11')
    np.savez_compressed(STEP1_3D_NPZ, 
                       data_x=data_x, data_y=data_y, y_type=y_type)

if __name__ == '__main__':
    main()
