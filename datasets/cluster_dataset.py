from torch.utils.data import Dataset


class ClusterDataset(Dataset):

    def __init__(self, data, labels, clusters):
        self.data = data
        self.labels = labels
        self.clusters = clusters

        if len(data) != len(labels) != len(clusters):
            raise Exception

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Return a raw_data point by given index.
        """
        return self.data[index], self.labels[index], self.clusters[index]
