#%%
import sys 
import os 
import os.path as osp 
import random 
from collections import Counter

from tqdm import tqdm 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

seed = 42
random.seed(seed)
np.random.seed(seed)



def action_binary(decimal_list: list) -> np.array: 
    
    
    label = list(map(int, list(np.binary_repr(0, width=12)))) # init

    for decimal in decimal_list:
        s = pd.Series(range(12)) 
        onehot = OneHotEncoder(sparse=False)
        activity_onehot = onehot.fit_transform(s.to_numpy().reshape(-1, 1)).astype('uint8')[decimal-1]

        label = np.bitwise_or(label, activity_onehot)

    return label




# print("Test", action_binary([2, 4, 5, 10, 7]))  


#%%




def single_attention_segment(label_decimal:int) : 
    # 행동 번호 리스트를 입력하면 이를 조합한 binary vector 반환 

    category_dict = {   0 : "nonattention",
                        1 : "neutral",
                        2 : "attention",
                        }

    action_scope = {    "nonattention" : action_binary([2, 4, 5, 7, 10]), 
                        "neutral" : action_binary([9]), 
                        "attention" : action_binary([ 12, 3, 11, 6, 8]),
                    }


    # ==== decimal to binary 
    binary = list(map(int, list(np.binary_repr(label_decimal, width=12))))

    
    quantity = [] 

    for key, value in action_scope.items():

        check_act = np.bitwise_and(action_scope[key],  binary)
        quantity.append(np.sum(check_act, axis=0))

    

    # ===== label decision
    if quantity[0] == 0:
        # no nonattention action 
        label = np.argmax(quantity, axis=0)

    else: 
        # there is any nonattention action 
        label = 0

    return category_dict[label], label



# print(single_attention_segment(label_decimal=8))











#%%
if __name__ == '__main__':

    rootDir = osp.join(".","anno")
    subject_list = sorted(os.listdir(rootDir))
    subjects = np.array(subject_list)
    np.random.shuffle(subjects)

    train_sub, test_sub = subjects[ : int(0.9*len(subjects))], subjects[int(0.9*len(subjects)) : ]





    # ======================== #
    #   Meta for Train & Val   #
    # ======================== #
    column_names = ["path", "category_name", "category", "training_split"]


    # ===== Flags 
    Paths = []
    Category_names = []
    Categories = []
    Training_splits = []

    
    for csv_name in tqdm(train_sub):
        subj_name = csv_name.split("_")[0]

        anno_df = pd.read_csv(osp.join(rootDir, csv_name))
        anno_df.pop("category2")


        # ===== decide label 
        label = [] 
        img_path = []
        for _, row in anno_df.iterrows():
            imgs = osp.join(subj_name, row["name"])

            if os.listdir(osp.join('multiple_dynamic_optical_flow_images', imgs )):
                # ==== if it's not empty directory 
                # 이미지가 없는 빈 디렉토리는 제외시킴 
                label.append(single_attention_segment(row['category1']))
                img_path.append(imgs)
            else: 
                continue

#        label = [ single_attention_segment(row['category1']) for idx, row in anno_df.iterrows()]
#        img_path = [osp.join(subj_name, row["name"]) for _, row in anno_df.iterrows()]
        
        label_arr = np.array(label).reshape(-1, 2)

        # ===== store them 
        Paths.extend(img_path)
        Category_names.extend(label_arr[:,0].tolist())
        Categories.extend(label_arr[:,1].tolist())

        
    print("Train: ", Counter(Category_names)) # (ref) https://stackoverflow.com/questions/2600191/how-can-i-count-the-occurrences-of-a-list-item

    # Train: val split
    for i, _ in enumerate(Paths):     

        if i % 10 == 0: 
            Training_splits.extend([False])

        else: 
            Training_splits.extend([True])



    print(len(Paths))
    print(len(Training_splits))

    
    # ===== Save the metadata in .csv 
    
    temp = []
    for list_item in [Paths, Category_names, Categories, Training_splits]:
        arr = np.array(list_item).reshape(-1,1)
        temp.append(arr)

    concate = np.hstack(temp)
    meta_df = pd.DataFrame(concate, columns = column_names)

    meta_df.to_csv(osp.join("attention_meta.csv"), index=True)



    # ======================== #
    #       Meta for test      #
    # ======================== #
    column_names = ["path", "category_name", "category"]

    Paths = []
    Category_names = []
    Categories = []
    Training_splits = []

    for csv_name in tqdm(test_sub):
        subj_name = csv_name.split("_")[0]

        anno_df = pd.read_csv(osp.join(rootDir, csv_name))
        anno_df.pop("category2")


        # ===== decide label 
        label = [] 
        img_path = []
        for _, row in anno_df.iterrows():
            imgs = osp.join(subj_name, row["name"])

            if os.listdir(osp.join('multiple_dynamic_optical_flow_images', imgs )):
                # ==== if it's not empty directory 
                # 이미지가 없는 빈 디렉토리는 제외시킴                 
                label.append(single_attention_segment(row['category1']))
                img_path.append(imgs)
            else: 
                continue

#        label = [ single_attention_segment(row['category1']) for idx, row in anno_df.iterrows()]
#        img_path = [osp.join(subj_name, row["name"]) for _, row in anno_df.iterrows()]
        
        label_arr = np.array(label).reshape(-1, 2)

        # ===== store them 
        Paths.extend(img_path)
        Category_names.extend(label_arr[:,0].tolist())
        Categories.extend(label_arr[:,1].tolist())        

    
    print("Test: ", Counter(Category_names)) # (ref) https://stackoverflow.com/questions/2600191/how-can-i-count-the-occurrences-of-a-list-item


    # ===== Save the metadata in .csv 
    
    temp = []
    for list_item in [Paths, Category_names, Categories]:
        arr = np.array(list_item).reshape(-1,1)
        temp.append(arr)

    concate = np.hstack(temp)
    meta_df = pd.DataFrame(concate, columns = column_names)

    meta_df.to_csv(osp.join("attention_meta_test.csv"), index=True)
