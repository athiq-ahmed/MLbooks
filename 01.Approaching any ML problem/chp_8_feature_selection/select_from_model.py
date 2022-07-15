# Select features based on feature coeffecients or the importance of features. features.
# If you use coefficients, you can select a threshold, and if the coefficient is above that threshold,
# you can keep the feature else eliminate it.

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

data = load_diabetes()
print(data)
X = data["data"]; print(X[0])
y = data["target"]; print(y[0])
col_names = data["feature_names"]; print(col_names)

model = RandomForestRegressor()
model.fit(X, y)

importances = model.feature_importances_
idxs = np.argsort(importances)
plt.title('Feature Importance')
plt.barh(range(len(idxs)), importances[idxs], align='center')
plt.yticks(range(len(idxs)),[col_names[i] for i in idxs] )
plt.xlabel("Random Forest Feature Importance")
plt.show()


###############################################################################################################
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

data = load_diabetes();print(data)
X = data["data"]; print(X[0])
y = data["target"]; print(y[0])
col_names = data["feature_names"]; print(col_names)

# Initiate the model
model = RandomForestRegressor()

# select from the model
sfm = SelectFromModel(estimator= model); print(sfm)
X_transformed = sfm.fit_transform(X, y); print(X_transformed[0])

# see which features are selected
print(col_names)
support = sfm.get_support();print(support)

# get feature names
print([x for x,y in zip(col_names, support) if y == True])