
import os 
import os.path as osp 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 



def read_excel(Path:str, drop_nan:bool=True, index_col=0) -> pd.core.frame.DataFrame: 
    df = pd.read_excel(Path,  index_col = index_col ) # (ref) https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

    if drop_nan:
        df = df.dropna(axis=0)  # drop rows(axis=0) including NaN

    return df




def read_csv(Path:str, drop_nan:bool=True, index_col=0) -> pd.core.frame.DataFrame: 
    df = pd.read_csv(Path, encoding='utf-8', index_col=index_col) # (ref) https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

    if drop_nan:
        df = df.dropna(axis=0)  # drop rows(axis=0) including NaN

    return df



def vis_dataFrame(df:pd.core.frame.DataFrame, title:str, xlabel:str, ylabel:str, fmt:str='.4g'):
    # (ref)https://datascienceparichay.com/article/get-column-names-as-list-in-pandas-dataframe/
    col_list = df.columns.values.tolist() 

    case = [] 
    for i in col_list: 
        arr = df[i].to_numpy().reshape(-1, 1) # vec to column array 
        case.append(arr)

    heat = np.hstack(case)

    # ==== heatmap by seaborn
    # (ref) https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (ref) https://stackoverflow.com/questions/29647749/seaborn-showing-scientific-notation-in-heatmap-for-3-digit-numbers
    ax = sns.heatmap(heat, annot=True, linewidths=.5, cmap="YlGnBu", fmt=fmt)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


