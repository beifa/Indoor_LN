import json
import os, glob
import pandas as pd
import numpy as np
from tqdm import tqdm



if __name__ == '__main__':
    PATH_DATA          = '/input'
    PATH_TEST_FOLDER   = '../input/test_folder'

    """
    find all bssid for current buildings
    data saved dict
        {name_buildings: [list all bssid]}
    """

    bssid = {}
    # unq_buildings = ['5d2709d403f801723c32bd39']
    for building in unq_buildings:
        print(building)
        tmp = []
        file_list = glob.glob(f'../input/train/train/{building}/*/*')
        for f in tqdm(file_list):         
            df = pd.read_parquet(f)
            wifi = df[df.columns_2 == 'TYPE_WIFI']
            tmp.append(wifi)
        df = pd.concat(tmp)
        value_counts = df['columns_4'].value_counts()
        top_bssid = value_counts[value_counts > 500].index.tolist()
        print(len(value_counts.index.tolist()))
        print('reduce',len(top_bssid))
        bssid[building] = top_bssid
        
    assert len(bssid) == 24, 'error size building'
    with open(os.path.join(PATH_DATA, "bssid_all.json"), "w") as f:
        json.dump(bssid, f)

    print('Correct')