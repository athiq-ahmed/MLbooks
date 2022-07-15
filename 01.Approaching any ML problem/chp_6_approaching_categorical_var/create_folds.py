import pandas as pd
from sklearn import model_selection

if __name__ =="__main__":
    df = pd.read_csv("Inputs/cat_train.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y=df.target
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_,'kfold']=f
    df.to_csv("Inputs/cat_train_folds.csv", index=False)
    print(df.kfold.value_counts())

for i in range(5):
    print(f"Fold-{i} \n{df[df.kfold == i].target.value_counts(normalize=True) * 100}")
    print(f"Fold-{i} \n{df[df.kfold == i].target.value_counts()}")
    print("*"*100)

