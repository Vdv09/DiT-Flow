from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from class_registry import class_registry


transform = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


@class_registry.add_to_registry("cifar10")
def get_loaders(batch_size = 128, num_workers = 4):
    train_data = datasets.CIFAR10(
        root = "./data",
        train = True,
        download = True,
        transform = transform
    )

    test_data = datasets.CIFAR10(
        root = "./data",
        train = False,
        download = True,
        transform = transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers
    )

    return train_loader, test_loader
