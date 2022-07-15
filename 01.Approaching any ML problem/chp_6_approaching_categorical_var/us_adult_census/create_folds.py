import pandas as pd
from sklearn import model_selection


df = pd.read_csv("Inputs/adult-training.csv")
df.columns = ["age","workclass","fnlwgt","education","education.num","marital.status","occupation",
              "relationship","race","sex","capital.gain","capital.loss","hours.per.week", "native.country",
              "income"]
df.head()

df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
kf = model_selection.StratifiedKFold(n_splits=5)
for f, (t,v) in enumerate(kf.split(X= df, y = df.income)):
    df.loc[v, "kfold"] = f
df.to_csv("Inputs/adult-training_folds.csv", index=False)
df.kfold.value_counts()
df.head()
df.columns

def fold_count(i):
    print(f"Fold-{i} \n{df[df.kfold == i].income.value_counts(normalize=True) * 100}")
    # print(f"Fold-{i} \n{df[df.kfold == i].income.value_counts()}")
    print("*"*100)

fold_count(0)
