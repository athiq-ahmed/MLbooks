from sklearn import datasets
from sklearn import model_selection
import numpy as np
import pandas as pd
import os

path = "Inputs"
filename = "mnist_train.csv"

def create_folds(data):
    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    num_bins = np.floor(1 + np.log2(len(data))).astype(int) # Sturge's rule
    data.loc[:, "bins"] = pd.cut(data["label"], bins=num_bins, labels=False)
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins)):
        data.loc[v_, "kfold"] = f
    data = data.drop("bins", axis=1)
    file = filename[:filename.find(".csv")]
    filename_kfold = file + "_kfolds" + ".csv"
    data.to_csv(os.path.join(path, filename_kfold))
    return data

df = pd.read_csv(os.path.join(path, filename))
df.head()
df.shape
df.label.value_counts().sort_index().plot(kind='bar')

# run the create folds code
df = create_folds(df)
df.head()
df.kfold.value_counts().sort_index().plot(kind='bar')




