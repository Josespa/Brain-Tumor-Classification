import numpy as np
import matplotlib.pyplot as plt
import itertools


def show_images(dataloader_images, labels_classes):
    images, labels = next(iter(dataloader_images))
    # Convert images to numpy arrays and denormalize
    images = (images.numpy().transpose((0, 2, 3, 1)) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
    # Create a grid of images
    num_images = len(images)
    rows = int(np.ceil(num_images / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(10, 10))
    # Plot images with labels
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.set_title(f'Label: {labels_classes[labels[i]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_training_history(result_data) -> None:
    """ Visualize training history in a graphic
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


def show_confusion_matrix(evaluating_class) -> None:
    """ Visualize confusion matrix
    :param evaluating_class: class to track evaluation performance
    :return: None
    """
    plt.figure(figsize=(10, 7))
    plt.grid(False)
    plt.imshow(evaluating_class.conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(evaluating_class.labels_classes))
    plt.xticks(tick_marks, [f"{key}" for key in evaluating_class.labels_classes], rotation=45)
    plt.yticks(tick_marks, [f"{key}" for key in evaluating_class.labels_classes])
    thresh = evaluating_class.conf_matrix.max() / 2.
    for i, j in itertools.product(range(evaluating_class.conf_matrix.shape[0]),
                                  range(evaluating_class.conf_matrix.shape[1])):
        plt.text(j, i, f"{evaluating_class.conf_matrix[i, j]}\n{evaluating_class.conf_matrix[i, j] /
                                                                np.sum(evaluating_class.conf_matrix) * 100:.2f}%",
                 horizontalalignment="center",
                 color="white" if evaluating_class.conf_matrix[i, j] > thresh else "black")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
