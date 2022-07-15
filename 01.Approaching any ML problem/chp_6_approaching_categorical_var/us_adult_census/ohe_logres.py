import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 1000)

"""
Split the dataset into train and valid
Concatenate into final dataset - Full data
fit onto the final dataset
transform on x_train and x_valid
Develop the model
"""
def run(fold):
    df = pd.read_csv("Inputs/adult-training_folds.csv")
    num_cols = ["fnlwgt",  "age",  "capital.gain",  "capital.loss",  "hours.per.week"]
    df = df.drop(num_cols, axis=1)
    target_mapping = {" <=50K":0, " >50K":1}
    df.loc[:, "income"] = df.income.map(target_mapping)

    features = [f for f in df.columns if f not in ("kfold", "income")]
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()
    full_data = pd.concat(
        [df_train[features],
        df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.income.values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"Accuracy = ", auc)

if __name__ == "__main__":
    for i in range(5):
        run(i)
###############################################################################################################################