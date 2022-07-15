""""
Target encoding is a technique in which you map each category in a given  feature to its mean target value,
but this must always be done in a cross-validated  manner.

It means that the first thing you do is create the folds, and then use those  folds to create target
encoding features for different columns of the data in the same  way you fit and predict the model on folds.

So, if you have created 5 folds, you have  to create target encoding 5 times such that in the end,
you have encoding for  variables in each fold which are not derived from the same fold.
And then when  you fit your model, you must use the same folds again.

Target encoding for unseen  test data can be derived from the full training data or can be an
average of all the 5  folds.

"""

import copy
import pandas as pd

from sklearn import preprocessing
import xgboost
from sklearn import metrics

def mean_target_encoding(data):
    df = copy.deepcopy(data)

    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]
    target_mapping = {" <=50K": 0, " >50K": 1}
    df.loc[:, "income"] = df.income.map(target_mapping)

    cat_cols = [c for c in df.columns if c not in num_cols and c not in ("kfold", "income")]
    features = [f for f in df.columns if f not in ("kfold", "income")]

    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])


    # list to store 5 validation dataframes
    encoded_dfs = []

    for fold in range(5):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        for col in cat_cols:
            mapping_dict = dict(df_train.groupby(col)["income"].mean())
            df_valid.loc[:, col+"_enc"] = df_valid[col].map(mapping_dict)
        encoded_dfs.append(df_valid)

    encoded_dfs = pd.concat(encoded_dfs, axis=0)
    return encoded_dfs

def run(fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [f for f in df.columns if f not in ("kfold", "income")]

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgboost.XGBClassifier(n_jobs=-1, max_depth=7)
    model.fit(x_train, df_train.income.values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"Accuracy = ", auc)


if __name__ == "__main__":
    df = pd.read_csv("Inputs/adult-training_folds.csv")
    df = mean_target_encoding(df)

    for i in range(5):
        run(i)

"""
When we use target  encoding, it’s better to use some kind of smoothing or adding noise in the encoded values. 

Scikit-learn has contrib repository which has target encoding with  smoothing, 
or you can create your own smoothing. 

Smoothing introduces some  kind of regularization that helps with not overfitting the model.

"""