"""
Greedy feature selection
1. Chosse a model
2. Select a loss/score function
3. Iteratively evaluate each feature and add it to the list of "good" features if it improves loss/score.

"""

import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    def evaluate_score(self, X, y):
        """
        This function evaluates model on data and returns Area Under ROC Curve (AUC)
        NOTE: We fit the data and calculate AUC on same data.
        WE ARE OVERFITTING HERE.
        But this is also a way to achieve greedy selection. k-fold will take k times longer.

        If you want to implement it in really correct way, calculate OOF AUC
        and return mean AUC over k folds.

        :param X: training data
        :param y: targets
        :return: overfitted area under the roc curve
        """

        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[: ,1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def _feature_selection(self, X, y):
        """
        This function does the actual greedy selection
        :param X: data, numpy array
        :param y: targets, numpy array
        :return: (best scores, best features)
        """

        good_features = []
        best_scores = []

        num_features = X.shape[1]

        while True:
            # Initialize best feature and best score
            this_feature = None
            best_score = 0

            # loop over all features
            for feature in range(num_features):
                if feature in good_features:
                    continue
                selected_features = good_features + [feature]
                xtrain = X[:, selected_features]
                score = self.evaluate_score(xtrain, y)
                if score > best_score:
                    this_feature = feature
                    best_score = score
                if this_feature != None:
                    good_features.append(this_feature)
                    best_scores.append(score)
                if len(best_scores) > 2:
                    if best_scores[-1] < best_scores[-2]:
                        break
        return best_scores[:-1], good_features[:-1]

    def __call__(self, X, y):
        scores, features = self._feature_selection(X, y)
        return X[:, features], scores

if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=100)
    X_transformed, scores = GreedyFeatureSelection()(X,y)

