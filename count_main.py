import argparse 
import sys 
import os 
import os.path as osp 

import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict

# =============== # 
# Argument by CLI # 
# =============== # 
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='data', help="Root directory ")  # to read 
parser.add_argument('--target', default='indiv', help="indiv or global ")


actVec_lens = [14, 6, 14, 13, 13, 12 ]


def read_excel(Path:str) -> pd.core.frame.DataFrame: 
    df = pd.read_excel(Path, index_col = 0 ) # (ref) https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
    df = df.dropna(axis=0)  # drop rows(axis=0) including NaN

    return df



#%%

def get_onehot(len_vec:int) -> np.array:
    # (ref) https://datamasters.co.kr/78
    s = pd.Series(range(len_vec))
    onehot = OneHotEncoder(sparse=False)

    activity_onehot = onehot.fit_transform(s.to_numpy().reshape(-1, 1)).astype('uint8')
    return activity_onehot


#%%

def action_count(path: str, act_onehot:np.array, target:str) -> OrderedDict:
    f_name = {  'indiv': 'Indiv_feature',
                'global' : 'Global_feature',
            }

    activity_dict = OrderedDict()

    csv_df = read_excel(path)
    
    Num, _ = csv_df.shape 

    for act_idx, label in enumerate(act_onehot):

        activity_dict[act_idx] = []


        for i in range(Num):
            anno = csv_df.iloc[i][6:].to_numpy()

#            print(label)
#            print(anno)
#            print()

            check_act = np.bitwise_and(label, anno)

            if np.array_equal(label, check_act): # (ref) https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html

                feature = csv_df[f_name[target]].iloc[i]
                activity_dict[act_idx].append(feature)            

    return activity_dict



#%% 
def signal_statistic(group:list)-> OrderedDict:
    signal_dict = OrderedDict()
    action_keys = group[0].keys()
    
    for key in action_keys:
        
        temp = [] 
        for i, _ in enumerate(group):
            temp.extend(group[i][key])
            
        # ===== action-to-brain_signal statistic 
        try:
            mean = sum(temp)/len(temp)
            signal_dict[key] = [mean, len(temp)]  # mean_signal, num_signal

        except ZeroDivisionError: 
            # it means, len(temp):=0
            signal_dict[key] = [0, 0]

    return signal_dict




def save_csv(col_name:list, item_list:list, path:str):
    
    stack = []
    for i in item_list:
        np_array = np.array(i).astype('float')
        h = np_array.shape[0] 

        if h < 14: 
            # Insert Nan 
            # Nan only works with float 
            # (ref) https://moonbooks.org/Articles/How-to-add-a-new-column-of-nan-values-in-an-array-matrix-with-numpy-in-python-/
            num_Nan = 14 - h
            col = np.zeros(num_Nan)
            col.fill(np.nan)  # column with Nam 
            np_array = np.insert(np_array, -2, col)  # make the length 14 by inserting Nan
        
        np_array = np_array.reshape(-1,1)

        stack.append(np_array)
    print( )
        
    concate = pd.DataFrame(np.hstack(stack), columns = col_name)
    concate.to_csv(path , index=True)





if __name__ == '__main__':


    args = parser.parse_args()

    rootDir = args.data_root
    targetFeature = args.target

    categoryPath = osp.join(rootDir, targetFeature)
    category_list = sorted(os.listdir(categoryPath ))


    Means = []
    Nums = [] 

    for idx, action_category in enumerate(category_list): 
        subj_files = sorted(os.listdir(osp.join(categoryPath, action_category )))
        actVec_len = actVec_lens[idx]
        act_onehot = get_onehot(actVec_len)

        print("")
        group = []
        for i, subj in tqdm(enumerate(subj_files), total=len(subj_files)): 

            if not subj.endswith('.xlsx'):
                continue

            annoPath = osp.join(categoryPath , action_category , subj)
            activity_dict = action_count(annoPath, act_onehot, targetFeature)

            group.append(activity_dict)

        # ==== Brain signal statistic for each category 
        signal_dict = signal_statistic(group)

        sigMean = [ val[0] for key, val in signal_dict.items()]
        sigNum = [ val[1] for key, val in signal_dict.items()]
        
        
        # ==== 
        Means.append(sigMean)
        Nums.append(sigNum)
    
    # Save as .csv 
    column_name = [i.split("_")[-1] for i in category_list]

    save_csv(column_name, Means, f"./{targetFeature}_Means.csv")
    save_csv(column_name, Nums, f"./{targetFeature}_Nums.csv")