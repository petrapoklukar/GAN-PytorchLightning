from schemes import DataInput
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

class MNIST_GAN_DataModule(LightningDataModule):

    def __init__(self, data_config:DataInput):
        super().__init__()
        self.data_config = data_config

        self.transform =  transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        # download
        MNIST(self.data_config.data_dir, train=True, download=True)
        MNIST(self.data_config.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_config.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.data_config.batch_size, 
                          shuffle=False, num_workers=self.data_config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.data_config.batch_size, 
                          shuffle=False, num_workers=self.data_config.num_workers)