import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

df = pd.read_csv(r'Inputs/mobile_train.csv')
print(df.head())
print(df.shape)

X = df.drop("price_range", axis=1).values
y = df.price_range

classifier = ensemble.RandomForestClassifier(n_jobs=-1)

param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth"   : [1, 2, 5, 7, 11, 15],
    "criterion"   : ['gini', 'entropy']
}

model = model_selection.RandomizedSearchCV(
    estimator=classifier,
    param_distributions = param_grid,
    n_iter=5,
    scoring='accuracy',
    verbose=10,
    n_jobs=-1,
    cv=5
)

model.fit(X,y)
print(f'Best score: {model.best_score_}')

print("Best parameters are: ")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print(f'\t{param_name}: {best_parameters[param_name]}')
