from schemes import DataInput
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from data_modules.CIFAR10PairDataset import CIFAR10PairDataset

CONTRASTIVE_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

CONTRASTIVE_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

GAN_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


class CIFAR10_contrastiveGAN_DataModule(LightningDataModule):

    def __init__(self, data_config:DataInput):
        super().__init__()
        self.data_config = data_config

    def prepare_data(self):
        # download
        CIFAR10(self.data_config.data_dir, train=True, download=True)
        CIFAR10(self.data_config.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit':
            # TODO: datasets should use the same train!!! 
            # GAN data
            self.gan_train_data = CIFAR10(self.data_config.data_dir, train=True, transform=GAN_TRAIN_TRANSFORMS)
            self.dims = tuple(self.gan_train_data[0][0].shape)
            # Contrastive data
            all_data = CIFAR10PairDataset(root=self.data_config.data_dir, train=True, transform=CONTRASTIVE_TRAIN_TRANSFORM)
            self.contrastive_train_data, self.contrastive_val_data = random_split(all_data, [45000, 5000])
            

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.gan_test_data = CIFAR10(self.data_config.data_dir, train=False, transform=GAN_TRAIN_TRANSFORMS)
            self.contrastive_test_data = CIFAR10PairDataset(root=self.data_config.data_dir, train=False, 
                                                            transform=CONTRASTIVE_TEST_TRANSFORM)
            self.dims = tuple(self.gan_test_data[0][0].shape)

    def train_dataloader(self):
        GAN_loader = DataLoader(
            self.gan_train_data,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
        )
        
        contrastive_loader = DataLoader(
            self.contrastive_train_data,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
        )
        
        loaders = {'GAN': GAN_loader, 'contrastive': contrastive_loader}
        return loaders

    def val_dataloader(self):
        return DataLoader(self.contrastive_val_data, batch_size=self.data_config.batch_size, 
                          shuffle=False, num_workers=self.data_config.num_workers)

    def test_dataloader(self):
        GAN_loader = DataLoader(
            self.gan_test_data,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
        )
        
        contrastive_loader = DataLoader(
            self.contrastive_test_data,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
        )
        
        loaders = {'GAN': GAN_loader, 'contrastive': contrastive_loader}
        return loaders