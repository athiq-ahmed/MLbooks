# Remove features with very low variance
from sklearn.feature_selection import VarianceThreshold
data = 'data.csv'
var_thresh = VarianceThreshold(threshold=0.1)
transformed_data = var_thresh.fit_transform(data)

# Remove features with high correlation.
# For calculating the correlation between different numerical features, you can use the Pearson correlation.
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data['target_names']

df = pd.DataFrame(X, columns=col_names)
df.head()

df.loc[:, "MedInc_Sqrt"] = df.MedInc.apply(np.sqrt)
df.head()

df.corr()

# Univariate feature selection
# Mutual Information, chi square(use for data which is non negative in nature), Anova F test

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper on different univariate feature selection models from scikit-learn.
        :param n_features: SelectPercentile if float else SelectKbest
        :param problem_type: Classification or Regression
        :param scoring: scoring function, string
        """
        if problem_type == "classification":
            valid_scoring = {
                "f_classif" : f_classif,
                "chi2" : chi2,
                "mututal_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression" : f_regression,
                "mututal_info_regression": mutual_info_regression
            }

        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")

        # if n_features is int, we use selectkbest #if n_features is float, we use selectpercentile #please note that it is int in both cases in sklearn

        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of feature")

        def fit(self, X, y):
            return self.selection.fit(X, y)

        def transform(self, X):
            return self.selection.transform(X)

        def fit_transform(self, X, y):
            return self.selection.fit_transform(X, y)

ufs = UnivariateFeatureSelection(n_features=0.1, problem_type="regression", scoring="f_regression")
ufs.fit(X,y)
X_transformed = ufs.transform(X)

