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
def run(fold):
    df = pd.read_csv("Inputs/adult-training_folds.csv");
    num_cols = ["fnlwgt",  "age",  "capital.gain",  "capital.loss",  "hours.per.week"]
    # df = df.drop(num_cols, axis=1)
    target_mapping = {" <=50K":0, " >50K":1}
    df.loc[:, "income"] = df.income.map(target_mapping)

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

    model = xgboost.XGBClassifier(n_jobs=-1)
    model.fit(x_train, df_train.income.values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"Accuracy = ", auc)

if __name__ == "__main__":
    for i in range(5):
        run(i)
#################################################################################################################