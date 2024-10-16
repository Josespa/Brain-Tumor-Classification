import torch
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt


def denormalize_as_numpy_arrays(images: torch.Tensor) -> np.array:
    """ Denormalizes images to numpy arrays.

    :param images: Images as tensors to be denormalized
    :return: Array of denormalized images
    """
    return (images.numpy().transpose((0, 2, 3, 1)) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)


def get_number_of_images_to_show(images: torch.Tensor) -> int:
    """ Returns numbers the length of the array or 16 as the max value.

    :param images: Images to show
    :return: Number of images to show
    """
    return len(images) if len(images) < 16 else 16


def get_number_of_rows(num_images: int) -> int:
    """ Returns number of rows for the grid.

    :param num_images: Number of images to show
    :return: Integer number of rows
    """
    return int(num_images / 4)


def show_images(batch_images: torch.utils, classes: list[str]) -> None:
    """ Show images in the batch.

    Show a maximum of 16 images per batch.

    :param batch_images: Iterable sample of data from dataset
    :param classes: List of classes in the dataset
    :return: None
    """
    images, labels = next(iter(batch_images))
    images = denormalize_as_numpy_arrays(images)
    num_images = get_number_of_images_to_show(images)
    number_of_rows = get_number_of_rows(num_images)
    fig, axes = plt.subplots(number_of_rows, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.set_title(f'Label: {classes[labels[i]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_training_history(result_data) -> None:
    """ Graph of the training history.

    Shows a graph with the Loss history and Accuracy history.
    
    :param result_data: Class to track training history
    :return: None
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(result_data.train_losses, label='Training Loss')
    plt.plot(result_data.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    plt.subplot(1, 2, 2)
    plt.plot(result_data.train_accuracies, label='Training Accuracy')
    plt.plot(result_data.val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy History')
    plt.tight_layout()
    plt.show()


def show_confusion_matrix(result_cm_display: sklearn.metrics) -> None:
    """ Heat map of the confusion matrix.

    Display the resulting confusion matrix.

    :param result_cm_display: Confusion matrix
    :return: None
    """
    result_cm_display.plot(cmap="PuBu")
    plt.show()
