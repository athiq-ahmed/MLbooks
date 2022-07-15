import pandas as pd

df = pd.read_csv("Inputs/cat_train.csv")
df.head()
df.shape

df.target.value_counts().plot(kind='bar')

# Convert categories to numbers - Label Encoding
df.ord_2.value_counts()
mapping = {
    "Freezing": 0,
    "Warm": 1,
    "Cold": 2,
    "Boiling Hot": 3,
    "Hot": 4,
    "Lava Hot": 5
}

df.loc[:,"ord_2"] = df.loc[:,"ord_2"].map(mapping)
df.ord_2.value_counts()

# Label Encoder using sklearn
from sklearn import preprocessing
df = pd.read_csv("Inputs/cat_train.csv")
df.loc[:, "ord_2"] = df.loc[:, "ord_2"].fillna("None")
lbl_enc = preprocessing.LabelEncoder()
df.loc[:,"ord_2"] = lbl_enc.fit_transform(df.ord_2.values)
df.ord_2.value_counts()

# Sparse arrays
import numpy as np
example = np.array(
    [
        [0,0,1],
        [1,0,0],
        [1,0,1]
     ]
)
print(example.nbytes)

from scipy import sparse
example = np.array(
    [
        [0,0,1],
        [1,0,0],
        [1,0,1]
     ]
)
sparse_example = sparse.csr_matrix(example)
print(sparse_example.data.nbytes)
print(sparse_example.data.nbytes
      +sparse_example.indptr.nbytes
      +sparse_example.indices.nbytes)

import numpy as np
from scipy import sparse
n_rows = 1000
n_cols = 100000
example = np.random.binomial(1, p=0.05, size=(n_rows, n_cols))
print(f"size of dense array: {example.nbytes}")
sparse_example = sparse.csr_matrix(example)
print(f"size of sparse array: {sparse_example.data.nbytes}")
full_size = (
        sparse_example.data.nbytes
        +sparse_example.indptr.nbytes
        +sparse_example.indices.nbytes)
print(f"Full size of sparse array: {full_size}")


import numpy as np
from scipy import sparse
example = np.array(
    [
        [0,0,0,0,1,0],
        [0,1,0,0,0,0],
        [1,0,0,0,0,0]
     ]
)
print(f"Size of dense array: {example.nbytes}")
sparse_example = sparse.csr_matrix(example)
print(f"Size of sparse array: {sparse_example.data.nbytes}")
full_size = (
    sparse_example.data.nbytes
    + sparse_example.indptr.nbytes
    + sparse_example.indices.nbytes
)
print(f"FUll size of sparse array: {full_size}")

# Use one hot encoding from sklearn to build dense and sparse arrays
import numpy as np
from sklearn import preprocessing
example = np.random.randint(1000, size=1000000)
ohe = preprocessing.OneHotEncoder(sparse=False) #Sparse equals flase to get dense array
ohe_example = ohe.fit_transform(example.reshape(-1,1))
print(f"Size of dense array: {ohe_example.nbytes}")
ohe = preprocessing.OneHotEncoder(sparse=True)
ohe_example = ohe.fit_transform(example.reshape(-1,1))
print(f"Size of sparse array: {ohe_example.data.nbytes}")
full_size = (
    ohe_example.data.nbytes
    + ohe_example.indptr.nbytes
    + ohe_example.indices.nbytes
)
print(f"Full size of sparse array: {full_size}")

# One more example
df = pd.read_csv("Inputs/cat_train_folds.csv")
features = [f for f in df.columns if f not in ["id", "target", "kfold"]]

for col in features:
    df.loc[:, col] = df[col].astype("str").fillna("NONE")

df_train = df[df.kfold != 0].reset_index(drop=True)
df_valid = df[df.kfold == 0].reset_index(drop=True)

df_train.shape
df_valid.shape

ohe = preprocessing.OneHotEncoder()
full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
full_data.shape
ohe.fit(full_data[features])
full_data.head()

x_train = ohe.transform(df_train[features])
x_valid = ohe.transform(df_valid[features])

df_train.head()
x_train

df_train.shape
x_train.shape

# Converting categorical variables into numerical
df = pd.read_csv("Inputs/cat_train.csv")
df[df.ord_2=="Boiling Hot"].shape
df.groupby(['ord_2'])["id"].count()
df.groupby(['ord_2'])["id"].transform("count")  # replace the column by its count
df.ord_2.value_counts()
df[["id", "ord_2"]].head()

df.groupby(["ord_1", "ord_2"])["id"].count().reset_index(name="count")
df[["id", "ord_1", "ord_2"]].head()

# Create new features from the categorical variables
df["new_feature"] = (
    df.ord_1.astype('str')
    +"_"
    +df.ord_2.astype('str')
)
df.new_feature

df["new_feature"] = (
    df.ord_1.astype('str')
    +"_"
    +df.ord_2.astype('str')
    + "_"
    + df.ord_3.astype('str')

)
df.new_feature

# Handling Nan values
df.ord_2.value_counts()
df.ord_2.fillna("None").value_counts()

# New value in the test dataset and not in the training dataset
import pandas as pd
from sklearn import preprocessing

train = pd.read_csv("Inputs/cat_train.csv")
test = pd.read_csv("Inputs/cat_test.csv")
train.shape
test.shape
train.head()
test.head()
test.loc[:, "target"] = -1
data = pd.concat([train, test]).reset_index(drop=True)
features = [x for x in train.columns if x not in ["id", "target"]]
for feat in features:
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = data[feat].fillna("NONE").astype("str").values
    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
train = data[data.target!=-1].reset_index(drop=True)
test = data[data.target==-1].reset_index(drop=True)

