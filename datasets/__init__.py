"""Dataset factory.

The folder convention is kept identical to MixerCSeg:
    <dataset_path>/train_img, <dataset_path>/train_lab,
    <dataset_path>/test_img,  <dataset_path>/test_lab
"""
import torch.utils.data
from .crack_dataset import CrackDataset


class CustomDatasetDataLoader:
    def __init__(self, args):
        self.args = args
        self.dataset = CrackDataset(args)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=not args.serial_batches,
            num_workers=int(args.num_threads),
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataloader:
            yield data


def create_dataset(args):
    return CustomDatasetDataLoader(args).load_data()
