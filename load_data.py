import os
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def find_parent_path() -> str:
    """ Find current directory
    :return: String with working directory
    """
    return os.path.dirname(os.path.realpath(__file__))


def path_with_datasets() -> str:
    """ The folder with both datasets (training and testing)
    - Example: Current path containing the datasets is
        /dataset/Training
        /dataset/Testing
    :return: String with the path in current directory that contains training and testing datasets
    """
    return 'dataset'


def get_dataset_path(dataset_category: str) -> str:
    """ Join the parent path to the dataset category and dataset category

    :param dataset_category: String could be Training or Testing
    :return: String with the directory for Training or Testing
    """
    parent_path = find_parent_path()
    dataset_path = path_with_datasets()
    return str(os.path.join(parent_path, dataset_path, dataset_category))


def get_models_directory(name_model_to_save: str) -> str:
    """ Returns the directory to save the trained model in.

    :param name_model_to_save: String with the name of the model.
    :return: String with the directory to save the trained model in.
    """
    return '/'.join(('models', name_model_to_save))


def load_dataset(dataset_category: str, image_resize: tuple) -> torchvision.datasets:
    """  Extract the images for each folder and use the folder's name as a class.

    Load the dataset and transform the images,
    transformation include resize, to tensor and normalization

    :param dataset_category: String with dataset's category, could be Training or Testing
    :param image_resize: Tuple with new size for images
    :return: Dataset with images
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    dataset_category = get_dataset_path(dataset_category)
    print(f'Extracting images from: {dataset_category}...')
    dataset_images = ImageFolder(dataset_category, transform=transform)
    return dataset_images


def load_batch(dataset_category: str, dataset: torchvision.datasets, batch_size: int) -> DataLoader:
    """ Load batches from dataset.

    Load that dataset into the DataLoader and can iterate through the dataset as needed,
    Retrieves samples in “mini-batches”.
    Uses shuffle data at every epoch in the training set to reduce model overfitting

    :param dataset_category: String with dataset's category, could be Training or Testing
    :param dataset: Dataset with images and classes from current directory
    :param batch_size: the number of data samples processed before the parameters are updated
    :return: iterable sample of data from dataset
    """
    if dataset_category == 'Training':
        shuffle = True
    elif dataset_category == 'Testing':
        shuffle = False
    else:
        print('Cannot happen')
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_classes(dataset: torchvision.datasets) -> list[str]:
    """ Get the classes from dataset.

    Uses the names of the folders in the dataset as a class

    :param dataset: Dataset with images and classes from current directory
    :return: List of classes in the dataset
    """
    return dataset.classes
