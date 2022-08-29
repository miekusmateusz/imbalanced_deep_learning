import json
import os

import numpy as np
from imblearn.metrics import geometric_mean_score
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def parse_configuration(file):
    if isinstance(file, str):
        with open(file) as json_file:
            return json.load(json_file)
    else:
        return file


def test_knn_classifier(configuration, data, labels):
    neigh = KNeighborsClassifier(n_neighbors=configuration['classifier']['k'])
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
                                                                        test_size=0.3,
                                                                        stratify=labels,
                                                                        random_state=20)
    print(configuration['classifier']['k'])
    neigh.fit(data_train, labels_train)
    y_pred = neigh.predict(data_test)

    acc = accuracy_score(labels_test, y_pred)
    gmean = geometric_mean_score(labels_test, y_pred)
    f1 = f1_score(labels_test, y_pred, average='macro')
    cfm = confusion_matrix(labels_test, y_pred)
    return acc, gmean, f1, cfm, neigh.classes_


def test_tree_classifier(configuration, data, labels):
    neigh = DecisionTreeClassifier(random_state=0, min_samples_split=4, min_samples_leaf=2, class_weight='balanced')
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
                                                                        test_size=0.3,
                                                                        stratify=labels,
                                                                        random_state=20)
    print(configuration['classifier']['k'])
    neigh.fit(data_train, labels_train)
    y_pred = neigh.predict(data_test)

    acc = accuracy_score(labels_test, y_pred)
    gmean = geometric_mean_score(labels_test, y_pred)
    f1 = f1_score(labels_test, y_pred, average='macro')
    cfm = confusion_matrix(labels_test, y_pred)
    return acc, gmean, f1, cfm, neigh.classes_


def stratifiedKCrossVKnn(configuration, data, label):
    neigh = KNeighborsClassifier(n_neighbors=configuration['classifier']['k'])
    skf = StratifiedKFold(n_splits=3)

    aggr_acc = []
    aggr_gmean = []
    aggr_f1 = []
    data = np.array(data)
    label = np.array(label)

    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        gmean = geometric_mean_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        aggr_acc.append(acc)
        aggr_gmean.append(gmean)
        aggr_f1.append(f1)
        cfm = confusion_matrix(y_test, y_pred)

    return np.mean(aggr_acc), np.mean(aggr_gmean), np.mean(aggr_f1), cfm, neigh.classes_


def stratifiedKCrossVDecisionTree(configuration, data, label):
    neigh = DecisionTreeClassifier(random_state=0, min_samples_split=4, min_samples_leaf=2, class_weight='balanced')
    skf = StratifiedKFold(n_splits=3)

    aggr_acc = []
    aggr_gmean = []
    aggr_f1 = []
    data = np.array(data)
    label = np.array(label)
    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        gmean = geometric_mean_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        aggr_acc.append(acc)
        aggr_gmean.append(gmean)
        aggr_f1.append(f1)
        cfm = confusion_matrix(y_test, y_pred)

    return np.mean(aggr_acc), np.mean(aggr_gmean), np.mean(aggr_f1), cfm, neigh.classes_


def plot_metrics(metrics, path, title, dir_path):
    isExist = os.path.exists(dir_path)
    if not isExist:
        os.makedirs(dir_path)
    plt.plot(metrics, label=title)
    plt.savefig(path)
    plt.clf()


def plot_conf_matrix(cfm, classes, path, dir_path):
    isExist = os.path.exists(dir_path)
    if not isExist:
        os.makedirs(dir_path)
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()
