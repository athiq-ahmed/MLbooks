"""
cross-validation is a step in the process of building a machine learning model which
helps us ensure that our models fit the data accurately and also ensures that we do not overfit.

"""

# Cross-validation
import pandas as pd
df = pd.read_csv("Inputs/datasets_918_1674_wineQualityReds.csv")
df.head()

df.quality.unique()
# a mapping dictionary that maps the quality values from 0 to 5
quality_mapping = {
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
}

df.loc[:, "quality"] = df.quality.map(quality_mapping)
df.head()
df.quality.unique()
df.shape

# Splitting the dataset
df = df.sample(frac=1).reset_index(drop=True)
df.head()

df_train = df.head(1000)
df_test = df.tail(599)

# Over fitting
""""
The model fits perfectly on the training set and performs poorly when it comes to  the test set.
In terms of loss, both training and test loss decreases as number of epochs, however at some point, test loss reaches its minimum and starts increasing 
even though the training loss decreases. We must stop training where the validation loss reaches its minimum value  
"""

from sklearn import tree
from sklearn import metrics

clf = tree.DecisionTreeClassifier(max_depth=7)

df.columns
cols = ['fixed.acidity', 'volatile.acidity', 'citric.acid',
        'residual.sugar', 'chlorides', 'free.sulfur.dioxide',
        'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]
clf.fit(df_train[cols], df_train.quality)

# Generate predictions
train_predictions = clf.predict(df_train[cols])
test_predictions = clf.predict(df_test[cols])

train_accuracy = metrics.accuracy_score(df_train.quality, train_predictions)
test_accuracy = metrics.accuracy_score(df_test.quality, test_predictions)

train_accuracy
test_accuracy

# Max_depth = 3; train_accuracy = 60.1 %, test_accuracy = 54.7%
# Max_depth = 7; train_accuracy = 78.5 %, test_accuracy = 59.9%

# Make a plot
from sklearn import tree
from sklearn import metrics

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

train_accuracies = [0.5]
test_accuracies = [0.5]

for depth in range(1,25):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    cols = ['fixed.acidity', 'volatile.acidity', 'citric.acid',
            'residual.sugar', 'chlorides', 'free.sulfur.dioxide',
            'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol'
            ]
    clf.fit(df_train[cols], df_train.quality)

    train_predictions = clf.predict(df_train[cols])
    test_predictions = clf.predict(df_test[cols])

    train_accuracy = metrics.accuracy_score(df_train.quality, train_predictions)
    test_accuracy = metrics.accuracy_score(df_test.quality, test_predictions)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
plt.plot(train_accuracies, label="train_accuracy")
plt.plot(test_accuracies, label="test_accuracy")
plt.legend(loc="upper left", prop={'size':15})
plt.xticks(range(0,26,5))
plt.xlabel("max_depth", size=20)
plt.ylabel("accuracy", size=20)
plt.show()

import pandas as pd
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt

# K Fold CV
if __name__ == "__main__":
    df = pd.read_csv('Inputs/train.csv')
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    # df.to_csv("Inputs/train_folds.csv", index=False)

df.head()
df.target.value_counts()
df.target.value_counts(normalize=True) * 100
df.kfold.value_counts(normalize=True)
df.groupby(['kfold', 'target'])['kfold'].count()


# Stratified KFold CV
df = pd.read_csv('Inputs/train.csv')
df['kfold'] = -1
df = df.sample(frac=1).reset_index(drop=True)
kf = model_selection.StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=df, y = df.target)):
    df.loc[val_, 'kfold'] = fold

df.head()
df.target.value_counts(normalize=True) * 100
df.kfold.value_counts(normalize=True)
df.groupby(['kfold', 'target'])['kfold'].count()  #The proportion of 1's are same across all kfolds

# Visualize the distribution of labels (wine dataset)
b = sns.countplot(x='quality', data=df)
b.set_xlabel("quality", fontsize=20)
b.set_ylabel("count", fontsize=20)
plt.show()

# Stratified Kfold for regression
import  numpy as np
import  pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    num_bins = np.floor(1 + np.log2(len(data))).astype(int) # Sturge's rule
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins)):
        data.loc[v_, "kfold"] = f
    data = data.drop("bins", axis=1)
    return data

if __name__ == "__main__":
    X, y = datasets.make_regression(n_samples=1500, n_features=100, n_targets=1)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
    df.head()
    df.loc[:, "target"] = y
    df = create_folds(df)

df.head()
df.kfold.value_counts()
