import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
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

df["date"].min(), df["date"].max()

df["country"].nunique()

df["store"].nunique()

df.groupby(["country", "store"]).agg({"num_sold": ["mean", "sum"]})\
    .sort_values(by=("num_sold", "mean"), ascending=False)


# Feature Engineering

df.head()
# cerating date features
def create_date_features(dataframe):
    dataframe['month'] = dataframe.date.dt.month
    dataframe['day_of_month'] = dataframe.date.dt.day
    dataframe['day_of_year'] = dataframe.date.dt.dayofyear
    dataframe['week_of_year'] = dataframe.date.dt.weekofyear
    dataframe['day_of_week'] = dataframe.date.dt.dayofweek
    dataframe['year'] = dataframe.date.dt.year
    dataframe["is_wknd"] = dataframe.date.dt.weekday // 4
    dataframe['is_month_start'] = dataframe.date.dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe.date.dt.is_month_end.astype(int)
    return dataframe

df = create_date_features(df)
df.head()

df.groupby(["store", "product", "month"]).agg({"num_sold": ["sum", "mean", "median", "std"]})


# Random Noise

# The lag features we will produce will be generated over sales,
# we add random noise to prevent over-learning while they are being produced.

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


df.groupby(["store", "product"])['num_sold'].mean()


# Lag/Shifted Features
df.sort_values(by=['store', 'product', 'date'], axis=0, inplace=True)

pd.DataFrame({"num_sold": df["num_sold"].values[0:10],
              "lag1": df["num_sold"].shift(1).values[0:10],  # birinci gecikme, shift ile alınıyor gecikme
              "lag2": df["num_sold"].shift(2).values[0:10],
              "lag3": df["num_sold"].shift(3).values[0:10],
              "lag4": df["num_sold"].shift(4).values[0:10]})


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "product"])['num_sold'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [366, 375, 386, 394, 424, 486, 527, 564, 637, 728])

get_stats(df)


# Rolling Mean Features
pd.DataFrame({"sales": df["num_sold"].values[0:10],
              "roll2": df["num_sold"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["num_sold"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["num_sold"].shift(1).rolling(window=5).mean().values[0:10]})

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "product"])['num_sold']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])


# Exponentially Weighted Mean Features
pd.DataFrame({"sales": df["num_sold"].values[0:10],
              "roll2": df["num_sold"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["num_sold"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["num_sold"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["num_sold"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["num_sold"].shift(1).ewm(alpha=0.1).mean().values[0:10]})


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "product"])['num_sold'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [366, 375, 386, 394, 424, 486, 527, 564, 637, 728]

df = ewm_features(df, alphas, lags)
get_stats(df)


# OHE
df = pd.get_dummies(df, columns=['store', 'product', 'day_of_week', 'month'])

get_stats(df)


# Converting sales to log(1+sales)
# logarithm operation, 1 because 0 cannot be logged, used to avoid some errors like this

df['num_sold'] = np.log1p(df["num_sold"].values)

get_stats(df)


# Modelling

# Custom Cost Functions
def smape(preds, target, epsilon=1e-10):
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)

    # Filter small values to avoid division by zero
    small_val_mask = denom < epsilon
    denom[small_val_mask] = epsilon

    smape_val = 200 * np.mean(num / denom)
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()  # LigtGBM veri yapısının içerisinde olan bağımlı değişkeni ifade ediyor
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


