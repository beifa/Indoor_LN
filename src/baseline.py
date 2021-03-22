import os, glob
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import lightgbm as lgb

if __name__ == '__main__':
    # train & test dir
    FILE_PATH_WIFI = 'input/wifi'
    # loads list files
    train_list = sorted(glob.glob(os.path.join(FILE_PATH_WIFI, 'train', '*')))
    test_list = sorted(glob.glob(os.path.join(FILE_PATH_WIFI, 'test', '*')))
    
    sbm = pd.read_csv('input/original/sample_submission.csv', index_col=0)

    pred = []

    for i, file in tqdm(enumerate(train_list)):
        # print(file)
        data = pd.read_csv(file)
        # tf = file.split('_')
        # tf[-1] = 'test.csv'
        # test_file = '_'.join(tf)
        # print(test_file)
        test_data = pd.read_csv(test_list[i])

        # all data not include x,y, floor and path file
        x_train = data.iloc[:,:-4]
        y_trainy = data.iloc[:,-3]
        y_trainx = data.iloc[:,-4]

        y_trainf = data.iloc[:,-2]

        modely = lgb.LGBMRegressor(
            n_estimators=15, num_leaves=90)
        modely.fit(x_train, y_trainy)

        modelx = lgb.LGBMRegressor(
            n_estimators=15, num_leaves=90)
        modelx.fit(x_train, y_trainx)

        modelf = lgb.LGBMClassifier(
            n_estimators=15, num_leaves=90)
        modelf.fit(x_train, y_trainf)
        
        test_predsx = modelx.predict(test_data.iloc[:,:-1]) # drop path
        test_predsy = modely.predict(test_data.iloc[:,:-1])
        test_predsf = modelf.predict(test_data.iloc[:,:-1])
        
        test_preds = pd.DataFrame(np.stack((test_predsf, test_predsx, test_predsy))).T
        """
        
        site_path_timestamp                                            loor   x          y
        
        5dc8cea7659e181adb076a3f_068e4f6926e78ff6d338d2cc_0000000000008 6 180.957362 141.994062
        5dc8cea7659e181adb076a3f_068e4f6926e78ff6d338d2cc_0000000003114 6 179.650218 142.849216
        5dc8cea7659e181adb076a3f_068e4f6926e78ff6d338d2cc_0000000005522 6 179.266679 142.369210
        
        """
        
        test_preds.columns = sbm.columns
        test_preds.index = test_data["site_path_timestamp"]
        test_preds["floor"] = test_preds["floor"].astype(int)
        pred.append(test_preds)

    # generate prediction file 
    all_preds = pd.concat(pred)
    all_preds = all_preds.reindex(sbm.index)
    all_preds.to_csv('submission/submission.csv')
    print('End')