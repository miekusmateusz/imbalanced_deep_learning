import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Visualizer:

    def __init__(self):
        self.scaler = StandardScaler()

    def reduce_data_to_2d_with_pca(self, data, y, directory_path, save_image_path, title):
        pca = PCA(n_components=2)
        X = pca.fit_transform(data)

        embedded_df = pd.DataFrame.from_dict({
            "x": X[:, 0],
            "y": X[:, 1],
            "color": y
        })
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=embedded_df, x="x", y="y", hue="color", palette="deep")
        plt.title(title)
        isExist = os.path.exists(directory_path)
        if not isExist:
            os.makedirs(directory_path)
        plt.savefig(save_image_path)
        plt.clf()
        plt.close()
