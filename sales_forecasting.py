import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

train = pd.read_csv("datasets/train.csv", parse_dates=["date"])
test = pd.read_csv("datasets/test.csv", parse_dates=["date"])

train
test

df = pd.concat([train, test], sort=False)
df.head()


# EDA

def get_stats(dataframe):
    return ("############### First 5 Line ###############", dataframe.head(),
            "############### Number of Values Owned ###############", dataframe.value_counts(),
            "############### Total Number of Observations ###############", dataframe.shape,
            "############### Variables Types ############### \n", dataframe.dtypes,
            "############### Total Number of Null Values ###############", dataframe.isnull().sum(),
            "############### Descriptive Statistics ###############", dataframe.describe().T
            )

get_stats(df)



