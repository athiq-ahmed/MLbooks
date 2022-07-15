import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

pd.set_option('display.width', 1000)


class Perceptron:
    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 1, 0)


s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('From URL:', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
print(df.head())

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y[:5]

y = np.where(y=='Iris-setosa', 0,1)
y[:5]

# extract sepal length and petal length
X = df.iloc[0:100, [0,2]].values
X[:5]

# plot data
plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length[cm]')
plt.ylabel('Petal length[cm]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

len(ppn.errors_)

plt.plot(range(1, len(ppn.errors_)+1, ppn.errors_, marker='o'))
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

