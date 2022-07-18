import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


pd.set_option('display.width', 1000)

"""
The perceptron receives the inputs of an example (x) and combines them with the bias unit (b) and weights (w) to compute the net input. 

The net input is then passed on to the threshold function, which generates a binary output of 0 or 1—the predicted class label of the example. 

During the learning phase, this output is used to calculate the error of the prediction and update the weights and bias unit.

"""

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


# A plot of the misclassification errors against the number of epochs
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

# visualize the decision boundaries for two-dimensional datasets:
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1 
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                            np.arange(x2_min, x2_max, resolution))         # grid arrays
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)                  # flatten the grid arrays
    lab = lab.reshape(xx1.shape)                                                      # reshaping with same dimensions as xx1 and xx2

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x = X[y==cl, 0],
            y = X[y==cl, 1], 
            alpha = 0.8,
            c = colors[idx],
            marker = markers[idx],
            label = f'Class{cl}',
            edgecolor = 'black'
        )

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length[cm]')
plt.ylabel('Petal length[cm]')
plt.legend(loc='upper left')
plt.show()

