import os, glob, json
import pandas as pd
import numpy as np
from tqdm import tqdm

PATH_ORIGINAL_DATA = '../input/original'
PATH_DATA          = '../input/'
PATH_TEST_FOLDER   = '../input/test_folder'

DATA_TYPES = ('TYPE_ACCELEROMETER',
                'TYPE_MAGNETIC_FIELD',
                'TYPE_GYROSCOPE',
                'TYPE_ROTATION_VECTOR',
                'TYPE_MAGNETIC_FIELD_UNCALIBRATED',
                'TYPE_GYROSCOPE_UNCALIBRATED',
                'TYPE_ACCELEROMETER_UNCALIBRATED',
                'TYPE_WIFI',
                'TYPE_BEACON',
                'TYPE_WAYPOINT' #target
                )

FLOOR_MAP = {"B2":-2, "B1":-1, "F1":0, "F2": 1, "F3":2, "F4":3, "F5":4, "F6":5, "F7":6,"F8":7, "F9":8,
             "1F":0, "2F":1, "3F":2, "4F":3, "5F":4, "6F":5, "7F":6, "8F": 7, "9F":8}



if __name__ == '__main__':

    """
    data to make wifi get from preprocessing.py
    
    """

    sub = pd.read_csv(os.path.join(PATH_ORIGINAL_DATA, 'sample_submission.csv'))
    buildings =  sub['site_path_timestamp'].apply(lambda x: pd.Series(x.split('_')))
    unq_buildings = buildings[0].value_counts().index.to_list()

    # bssid.py
    with open(os.path.join(PATH_DATA, 'bssid_all.json')) as f:
        bssid = json.load(f)

    ############TRAIN

    # unq_buildings = ['5d2709d403f801723c32bd39']
    for building in unq_buildings:
        print(building)
        
        file_list = glob.glob(f'../input/train/train/{building}/*/*')

        box = []
        building_bssid = bssid[building]
        for f in tqdm(file_list):
            # ../input/indoor-data/train/train/5a0546857ecc773753327266/B1/5e15730aa280850006f3d005.parquet
            # take B1
            f_floor = f.split(os.path.sep)[-2]
            floor = FLOOR_MAP[f_floor]
            df = pd.read_parquet(f) 

            waypoint = df[df.columns_2 == 'TYPE_WAYPOINT']
            wifi     = df[df.columns_2 == 'TYPE_WIFI']
        
            new_df =pd.DataFrame()
            for time, g in wifi.groupby('columns_1'):
                tmp = []
                for k in waypoint.columns_1.values:
                    dif = abs(int(time) - int(k))
                    tmp.append(dif)
                """
                tmp
                    0, 1, 2, .... 8
                """
                min_idx = np.argmin(tmp)
                g = g.drop_duplicates(subset='columns_4')
                g_tmp = g.iloc[:, 3:5]
                new_df = g_tmp.set_index('columns_4').reindex(building_bssid).replace(np.nan, -999).T
                new_df['x'] = float(waypoint.iloc[min_idx].columns_3)
                new_df['y'] = float(waypoint.iloc[min_idx].columns_4)
                new_df['floor'] = floor
                new_df['path'] = g.columns_11.values[0]
                box.append(new_df)
        """
        time make     
            5d2709d403f801723c32bd39 :
                                    all [21:50<00:00,  3.63s/it] 10027 rows × 2143 columns
                                    >100 [11:32<00:00,  1.92s/it] 10027 rows × 1111 columns
                                    >500 [06:27<00:00,  1.07s/it] 10027 rows × 568 columns
        """        
        # all wifi for building
        building_df = pd.concat(box)
        building_df.reset_index(drop=True).to_csv(os.path.join(PATH_TEST_FOLDER, building+"_train.csv"), index=False)
    #     break

    #################TEST

    # from submision 
    for building, id_ in buildings.groupby(0):
        # go each building
        # id_ all data by building
        print(building)
        files = glob.glob(f'../input/test/test/{building}/*')
        building_bssid = bssid[building]
        print('bssid',  len(building_bssid))
        tmp = []
        for idd, g in id_.groupby(1):  
            # load building + idd
            df = pd.read_parquet(f'../input/test/test/{building}/{idd}.parquet')
            df = df[df.columns_2 == 'TYPE_WIFI']
            wifi_time = df.groupby('columns_1').count().index
            for time in g[2].values:
                # eq wifi time from submission
                dif = abs(wifi_time.astype(int) - int(time)) 
                # find nearest time point for all
                min_t = wifi_time[dif.values.argmin()]
                block = df[df.columns_1 == min_t].drop_duplicates(subset='columns_4')
                """
                0000000015640    189
                0000000017596    189
                0000000019554    185
                0000000013660    185
                ...
                0000000052342     57
                0000000050468     57           
                
                len = 35 uniques times           
                all line in df = 4597
                after filter by min time we get only 137 row and one min time            
                this neares by time point time submit and time in wifi for current site after naked in this features            
                """            
                block = block.set_index('columns_4')['columns_5'].reindex(building_bssid).fillna(-999)
                block['site_path_timestamp'] = g.iloc[0,0] + "_" + g.iloc[0,1] + "_" + time
                tmp.append(block)
  
        print('End building') 
        to_save = pd.concat(tmp, axis=1).T
        to_save.to_csv(os.path.join(PATH_TEST_FOLDER, building + "_test.csv"), index = False)
        # break