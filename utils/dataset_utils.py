import torch
from torch.utils.data import TensorDataset


def initialize_datasets(data_holder):
    train_dataset = TensorDataset(torch.Tensor(data_holder.data_train), torch.Tensor(data_holder.labels_train))
    train_dataset.train_data = torch.Tensor(data_holder.data_train)
    train_dataset.train_labels = torch.Tensor(data_holder.labels_train)
    train_dataset.train = True
    train_dataset.train_clusters = data_holder.train_clusters
    train_dataset.train_class_cluster_dict = data_holder.train_class_cluster_dict

    test_dataset = TensorDataset(torch.Tensor(data_holder.data_test), torch.Tensor(data_holder.labels_test))
    test_dataset.test_data = torch.Tensor(data_holder.data_test)
    test_dataset.test_labels = torch.Tensor(data_holder.labels_test)
    test_dataset.train = False
    test_dataset.test_clusters = data_holder.test_clusters
    test_dataset.test_class_cluster_dict = data_holder.test_class_cluster_dict

    whole_dataset = TensorDataset(torch.Tensor(data_holder.data), torch.Tensor(data_holder.labels))
    whole_dataset.data = torch.Tensor(data_holder.data)
    whole_dataset.labels = torch.Tensor(data_holder.labels)

    return train_dataset, test_dataset, whole_dataset
