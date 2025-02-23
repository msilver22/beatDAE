import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning as L

data_dir = './data'
batch_size = 32

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir=data_dir, batch_size=batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_set, self.val_set = random_split(mnist_full, [55000, 5000])
        if stage == 'test' or stage is None:
            self.test_set = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
