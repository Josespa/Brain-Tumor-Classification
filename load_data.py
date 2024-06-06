import os
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from plot import show_images


def find_parent_path():
    """

    :return:
    """
    return os.path.dirname(os.path.realpath(__file__))


def get_dataset_path(dataset_category):
    """

    :param dataset_category:
    :return:
    """
    parent_path = find_parent_path()
    return str(os.path.join(parent_path, 'dataset', dataset_category))


def load_dataset(dataset_category: str, image_resize: tuple) -> torchvision.datasets:
    """

    :param dataset_category:
    :param image_resize:
    :return:
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    dataset_category = get_dataset_path(dataset_category)
    print(f"Extracting images from: {dataset_category}...")
    dataset_images = ImageFolder(dataset_category, transform=transform)
    return dataset_images


def load_batch(dataset_category: str, dataset, target_batch_size):
    """

    :param dataset_category:
    :param dataset:
    :param target_batch_size:
    :return:
    """
    if dataset_category == "Training":
        shuffle = True
    elif dataset_category == "Testing":
        shuffle = False
    else:
        print("Can't happen")
        shuffle = False
    return DataLoader(dataset, batch_size=target_batch_size, shuffle=shuffle)


if __name__ == '__main__':
    batch_size = 8
    train_dataset = load_dataset("Training", (64, 64))
    train_dataloader = load_batch("Training", train_dataset, 8)
    show_images(train_dataloader, train_dataset.classes)
