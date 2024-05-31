import os
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def find_parent_path():
    """
    :param: None
    :return: current directory file
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path


def load_dataset(dataset_directory: str, target_image_size: tuple) -> torchvision.datasets:
    """
    :param dataset_directory: path for dataset
    :param target_image_size:
    :return:
    """
    parent_path = find_parent_path()
    transform = transforms.Compose(
        [
            transforms.Resize(target_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    dataset_directory = str(os.path.join(parent_path, dataset_directory))
    dataset_images = ImageFolder(dataset_directory, transform=transform)
    print(f"Extracting data from {dataset_directory}")
    return dataset_images


def show_images():
    print(3)

from torch import nn
if __name__ == '__main__':
    image_size = (64, 64)
    batch_size = 9
    dataset_path = "dataset/Training"
    train_dataset = load_dataset(dataset_path, image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(type(nn.CrossEntropyLoss()))
    show_images()
