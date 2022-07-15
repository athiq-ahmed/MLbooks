import itertools
import pandas as pd
from sklearn import preprocessing

import xgboost
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 1000)

"""
fit and transform on the entire dataset
split into df_train and df_valid

"""

def feature_engineering(df, cat_cols):
    """
    :param df:the pandas dataframe with train/test data
    :param cat_cols:list of categorical columns
    :return: dataframe with new features

    This will create all 2-combinations of values in this list
    for example:
    list(itertools.combinations([1,2,3], 2)) will return [(1, 2), (1, 3), (2, 3)]

    """
    combi = list(itertools.combinations(cat_cols,2))
    for c1, c2 in combi:
        df.loc[:, c1+"_"+c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

def run(fold):
    df = pd.read_csv("Inputs/adult-training_folds.csv");
    num_cols = ["fnlwgt",  "age",  "capital.gain",  "capital.loss",  "hours.per.week"]
    # df = df.drop(num_cols, axis=1)
    target_mapping = {" <=50K":0, " >50K":1}
    df.loc[:, "income"] = df.income.map(target_mapping)

    cat_cols = [c for c in df.columns if c not in num_cols and c not in ("kfold", "income")]
    df = feature_engineering(df, cat_cols)

    # print(df.head())
    features = [f for f in df.columns if f not in ("kfold", "income")]

    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgboost.XGBClassifier(n_jobs=-1, max_depth=7)
    model.fit(x_train, df_train.income.values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"Accuracy = ", auc)

if __name__ == "__main__":
    for i in range(5):
        run(i)
#################################################################################################################