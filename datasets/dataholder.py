from collections import Counter

import numpy as np
import pandas as pd
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from datasets.undersampling_method_enum import UnderSamplingMethod


class DataHolder:

    def __init__(self, dataset_path, majority_labels, k_param, undersample: bool, dataset_name,
                 undersample_method=UnderSamplingMethod.TOMEK_LINKS, ):

        self.dataset_name = dataset_name
        self.test_weights = None
        self.train_weights = None
        self.test_clusters = None
        self.train_clusters = None
        self.test_class_cluster_dict = {}
        self.train_class_cluster_dict = {}

        self.weights = None
        self.labels_test = None
        self.labels_train = None
        self.data_test = None
        self.data_train = None

        self.k_param = k_param

        self.dataset = pd.read_csv(dataset_path, sep=';')
        self.data = self.dataset.iloc[:, :-1].to_numpy()

        self.labels = self.dataset.iloc[:, -1].to_numpy()
        self.majority_labels = majority_labels

        self.undersample_method = undersample_method
        if undersample:
            self.undersample_dataset()

        self.train_test_split_dataset()
        self.calculate_global_imbalance_weights()

        self.cluster_with_k_means(train=True)
        self.cluster_with_k_means(train=False)

    def undersample_dataset(self):
        if self.undersample_method == UnderSamplingMethod.TOMEK_LINKS:
            undersampling_method = TomekLinks(sampling_strategy=self.majority_labels)

        elif self.undersample_method == UnderSamplingMethod.EDITED_NEAREST_NEIGHBOUR:
            undersampling_method = EditedNearestNeighbours(sampling_strategy=self.majority_labels)

        elif self.undersample_method == UnderSamplingMethod.RANDOM_UNDERSAMPLING:
            d = dict(Counter(self.labels))
            class_dist = dict()
            for (key, value) in d.items():
                if key not in self.majority_labels:
                    class_dist[key] = value
            minor_label_with_max_val = max(class_dist, key=class_dist.get)
            for min_l in self.majority_labels:
                if self.dataset_name == 'glass':
                    class_dist[min_l] = 2 * class_dist[minor_label_with_max_val]
                else:
                    class_dist[min_l] = class_dist[minor_label_with_max_val]

            undersampling_method = RandomUnderSampler(sampling_strategy=class_dist)
        else:
            print("Wrong method specified. Not resampled.")
            raise Exception
        self.data, self.labels = undersampling_method.fit_resample(self.data, self.labels)

    def train_test_split_dataset(self, test_size: float = 0.2, random_state: int = 42):

        self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(self.data, self.labels,
                                                                                                test_size=test_size,
                                                                                                stratify=self.labels,
                                                                                                random_state=random_state)

    def calculate_global_imbalance_weights(self):
        classes_counter = Counter(self.labels_train)
        weights = {c: 1 / v for c, v in classes_counter.items()}
        self.train_weights = {c: weights[c] / sum(weights.values()) for c in classes_counter.keys()}

        classes_counter = Counter(self.labels_test)
        weights = {c: 1 / v for c, v in classes_counter.items()}
        self.test_weights = {c: weights[c] / sum(weights.values()) for c in classes_counter.keys()}

    def cluster_with_k_means(self, train: bool):

        if train == True:
            data = self.data_train
            lbls = self.labels_train
        else:
            data = self.data_test
            lbls = self.labels_test

        dataset_clusters = np.empty(len(data))
        labels_set = set(lbls)
        label_to_indices = {label: np.where(lbls == label)[0] for label in labels_set}
        counter = 0
        for label, indices in label_to_indices.items():
            kmeans = KMeans(n_clusters=self.k_param, random_state=0)
            one_class_data = data[indices]
            kmeans.fit(one_class_data)

            # add counter to the predicted clusters
            predicted_clusters = np.array(kmeans.labels_)
            if train == True:
                self.train_class_cluster_dict[label] = []
            else:
                self.test_class_cluster_dict[label] = []

            for i in range(self.k_param):
                predicted_clusters = np.where(predicted_clusters == i, i + counter * self.k_param, predicted_clusters)
                if train == True:
                    self.train_class_cluster_dict[label].append(i + counter * self.k_param)
                else:
                    self.test_class_cluster_dict[label].append(i + counter * self.k_param)

            for idx in range(len(predicted_clusters)):
                predicted_cluster = predicted_clusters[idx]
                index = indices[idx]
                dataset_clusters[index] = predicted_cluster

            counter += 1
        if train == True:
            self.train_clusters = dataset_clusters
        else:
            self.test_clusters = dataset_clusters
