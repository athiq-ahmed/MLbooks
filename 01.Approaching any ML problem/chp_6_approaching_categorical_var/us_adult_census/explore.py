import pandas as pd

df = pd.read_csv("Inputs/adult-training.csv")
df.head()
df.shape
df.columns = ["age","workclass","fnlwgt","education","education.num","marital.status","occupation",
"relationship","race","sex","capital.gain","capital.loss","hours.per.week", "native.country","income"]
df.head()

df.income.value_counts()
df.income.value_counts(normalize=True)*100
