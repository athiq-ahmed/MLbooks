# Supervised and Unsupervised learning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn import manifold

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)

data = datasets.fetch_openml(
                'mnist_784',
                version=1,
                return_X_y=True)
data

pixel_values, targets = data
targets = targets.astype(int)

pixel_values.shape
targets.shape

single_image = pixel_values[1, :].reshape(28, 28)
plt.imshow(single_image, cmap='gray')
plt.show()

tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values[:3000, :])

tsne_df = pd.DataFrame(np.column_stack((transformed_data, targets[:3000])),
                       columns=["x", "y", "targets"])

tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)
tsne_df.head()

grid = sns.FacetGrid(tsne_df, hue="targets", size=8)
grid.map(plt.scatter, "x", "y").add_legend()

