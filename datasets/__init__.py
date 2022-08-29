from torch.utils.data import DataLoader

from datasets.dataset_wrapper import DatasetWrapper
from datasets.undersampling_method_enum import UnderSamplingMethod


def get_dataloader(dataset: DatasetWrapper, batch_size):
    return DataLoader(dataset, batch_size=batch_size)
