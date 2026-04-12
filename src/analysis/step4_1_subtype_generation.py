import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from project_paths import STEP3_CLUSTER_CSV, STEP4_SUBTYPE_CSV, ensure_dir

def generate_subtypes(df_data):
    cls_list=[]
    label_list = []
    for ID in tqdm(df_data['PATID'].unique()):
        tmp = df_data[df_data['PATID']==ID]
        cls_list += [str(tmp['cluster'].unique())]
        label_list += [int(tmp['label'].unique())]
    df_subtypes = pd.DataFrame(cls_list)
    df_subtypes.rename(columns={0:'cls_pattern'}, inplace=True)
    df_subtypes['PATID'] = df_data['PATID'].unique()
    df_subtypes = pd.merge(df_subtypes, df_data[['PATID', 'label']].drop_duplicates())
    return df_subtypes

def main():
    # 1. load data
    df_data = pd.read_csv(STEP3_CLUSTER_CSV)
    
    # 2. generate subtypes
    df_subtypes = generate_subtypes(df_data)

    df_tmp = df_data.sort_values(
        by=['PATID', 'subseq_PATID'],
        key=lambda col: col.str.split('_').str[-1].astype(int) if col.name == 'subseq_PATID' else col
    ).drop_duplicates(subset=['PATID'], keep='last')

    df_subtypes = pd.merge(df_subtypes, pd.concat([df_tmp['PATID'], df_tmp.iloc[:, 3:]], axis=1), on='PATID', how='left')
    
    ensure_dir(STEP4_SUBTYPE_CSV.parent)
    df_subtypes.to_csv(STEP4_SUBTYPE_CSV, index=False)


if __name__ == "__main__":
    main()
