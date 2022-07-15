import pandas as pd

s = pd.date_range('2020-01-06', '2020-01-10', freq = '10H').to_series()
s

# Create some features based on datetime
features = {
    "dayofweek": s.dt.dayofweek.values,
    "dayofyear": s.dt.dayofyear.values,
    "hour": s.dt.hour.values,
    "is_leap_year": s.dt.is_leap_year.values,
    "quarter": s.dt.quarter.values,
    "weekofyear": s.dt.weekofyear.values
}
features

def generate_featues(df):
    # create a bunch of features using datetime column
    df.loc[:, "year"] = df['date'].dt.year
    df.loc[:, "weeofyear"] = df['date'].dt.weekofyear
    df.loc[:, "month"] = df['date'].dt.month
    df.loc[:, "dayofweek"] = df['date'].dt.dayofweek
    df.loc[:, "weekend"] = (df['date'].dt.weekday >= 5).astype(int)

    # create an aggregate dictionary
    aggs = {}
    aggs["month"] = ["nunique", "month"]
    aggs["weekofyear"] = ["nunique", "month"]
    aggs["num1"] = ["sum", "max", "min", "mean"]
    aggs["customerid"] = ["size"]
    aggs["customerid"] = ["nunique"]

    # we group by customer_id and calculate the aggregates
    agg_df = df.groupby('customer_id').agg(aggs)
    agg_df = agg_df.reset_index(drop=True)
    return agg_df

# Time series problem with list of values
import numpy as np
feature_dict= {}
feature_dict['mean'] = np.mean(x)
feature_dict['max'] = np.max(x)
feature_dict['min'] = np.min(x)
feature_dict['std'] = np.std(x)
feature_dict['var'] = np.var(x)
feature_dict['ptp'] = np.ptp(x) #peak to peak

# percentile features
feature_dict['percentile_10'] = np.percentile(x, 10)
feature_dict['percentile_60'] = np.percentile(x, 60)
feature_dict['percentile_90'] = np.percentile(x, 90)

# quantile features
feature_dict['quantile_5'] = np.quantile(x, 5)
feature_dict['quantile_95'] = np.quantile(x, 95)
feature_dict['quantile_99'] = np.quantile(x, 99)

from tsfresh.feature_extraction import feature_calculators as fc

# tsfresh based features
feature_dict['abs_energy'] = fc.abs_energy(x)
feature_dict['count_above_mean'] = fc.count_above_mean(x)
feature_dict['count_below_mean'] = fc.count_below_mean(x)
feature_dict['mean_abs_change'] = fc.mean_abs_change(x)
feature_dict['mean_change'] = fc.mean_change(x)

import numpy as np
# generate a random dataframe with 2 columns and 100 rows
df = pd.DataFrame(
    np.random.rand(100,2),
    columns = [f"f_{i}" for i in range(1,3)]
)

df.head()

from sklearn import preprocessing
pf = preprocessing.PolynomialFeatures(
    degree=2,
    interaction_only=False,
    include_bias=False
)
pf.fit(df) # fit to features
poly_feats = pf.transform(df) # create polynomial features
# create a dataframe with all features
num_feats = poly_feats.shape[1]
df_transformed = pd.DataFrame(
    poly_feats,
    columns = [f"f_{i}" for i in range(1, num_feats +1)]
)
df_transformed.head()
df_transformed.plot(kind='bar')

# create bins of numerical columns
df["f_bins_10"] = pd.cut(df["f_1"], bins=10, labels=False) # 10 bins
df["f_bins_100"] = pd.cut(df["f_1"], bins=100, labels=False) # 10 bins
df.head()

df.loc[:, "f_3"] = np.random.randint(0, 10000, df.shape[0])
df.head()
df['f_3'].plot(kind='bar')
df['f_3'].hist(bins=10)

df.f_3.var()
df['f_3'] = df.f_3.apply(lambda x: np.log(1+x)).var()
df.head()

# KNN - Missing Imputation
import numpy as np
from sklearn import impute
# create a random numpy array with 10 samples and 6 features and values ranging from 1 to 15
X = np.random.randint(1,15,(10,6));X
X = X.astype(float);X
X.ravel()[np.random.choice(X.size, 10, replace=False)] = np.nan; X
knn_imputer = impute.KNNImputer(n_neighbors=2)
knn_imputer.fit_transform(X)

