#%% 
import argparse 
import sys 
import os 
import os.path as osp 
from pathlib import Path 
from glob import glob 

import numpy as np 
import pandas as pd 
from tqdm import tqdm 

import utils 



# =============== # 
# Argument by CLI # 
# =============== # 
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='anno', help="anno .csv dir ")  # to read 
parser.add_argument('--category', default='./data/indiv', help="labels by categories")
parser.add_argument('--output', default='re_anno', help="save dir for re-annotated .csv")




#%% 
def get_categories_by_target(categoryDir_path:str, category_list:list, target:str) -> list:
    path_list = []

    for category in category_list:
        # (ref) https://wikidocs.net/83
        path = glob(osp.join(categoryDir_path, category, f"{target}*"))

        path_list.append(path)

    return path_list




def get_action_vecs(df:pd.core.frame.DataFrame) -> list: 

    vec_list = []
    num, _ = df.shape 
    
    for i in range(num):
        anno_vec = df.iloc[i][6:].tolist()
        anno_vec = anno_vec[:-2]    # delete the last two default labels 
                                    # {no_action, default}

        vec_list.append(anno_vec)  


    return vec_list



def get_action_decimal(act_list:list) -> np.array:
    np_arr = np.asarray(act_list, dtype=np.int)

    num_instance, vec_dim = np_arr.shape


    diff = 12 - vec_dim 

    if diff:
        # ===== make the vector in 12-D
        fill_zero = np.zeros([num_instance, diff], dtype=np.int)
        np_arr = np.concatenate((np_arr, fill_zero), axis=1) # (ref) https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html


    # ===== binary to decimal 
    # e.g., [0 0 1 0] -> [2]
    action_list = [ binary_to_decimal(np_arr[i]) for i in range(num_instance) ]
    action_decimal = np.asarray(action_list, dtype=np.int)

    return action_decimal




def binary_to_decimal(binary_vec:np.array) -> int:
    # (ref) https://www.geeksforgeeks.org/python-binary-list-to-integer/

    res = 0 
    for ele in binary_vec:
        res = (res << 1) | ele

    return res 
    





#%% 
if __name__ == '__main__':
    args = parser.parse_args()

    annoDir = args.data_root
    categoryDir = args.category
    saveDir = args.output

    savePath = Path(saveDir)
    savePath.mkdir(parents=True, exist_ok=True) 

    annoCSV_list = sorted(os.listdir(annoDir))
    category_list = sorted(os.listdir(categoryDir))

    
    # ===== Read .csv 
    for annoCSV in annoCSV_list:
        print(annoCSV)

        Path_anno = osp.join(annoDir, annoCSV)
        df_anno = utils.read_csv(Path = Path_anno, index_col=False)

        target = annoCSV.split("_")[0]

        # ===== categories by target 
        path_list = get_categories_by_target(categoryDir, category_list, target)

        label_by_category =[]

        for path in path_list:
            print(path[0])
            df_excel = utils.read_excel(Path=path[0], drop_nan=True, index_col=0)


            # ===== Get the action vector 
            vec_list = get_action_vecs(df_excel)

            action_decimal = get_action_decimal(vec_list)
            action_decimal = action_decimal.reshape(-1, 1)

            label_by_category.append(action_decimal)


        # ===== Adding new columns to the DataFrame 
        # (ref) https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.add.html
        column_names = ["name", "indiv_feature", "global_feature", "motion",
                        "category1", "category3"]

        category_concate = np.hstack(label_by_category)
        new_df = pd.DataFrame(np.hstack([df_anno.to_numpy(), category_concate]), columns = column_names )


        new_df.to_csv(osp.join(saveDir, annoCSV), index=False)

