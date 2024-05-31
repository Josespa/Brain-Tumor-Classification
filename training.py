import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_data import load_dataset
from cnn_model import CNN


class TrainHistory:
    """ Save information during training
    """
    epochs = int
    device = str
    best_loss = float
    val_loss = float
    train_losses = list
    val_losses = list
    train_accuracies = list
    val_accuracies = list
    trained_model = str

    def __init__(self, num_epochs, new_model_path):
        self.epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_loss = float('inf')
        self.val_loss = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.trained_model = new_model_path


def plot_training_history(result_data: TrainHistory) -> None:
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


def training(cnn_model: nn.Module, training_class: TrainHistory, train_dataloader: torch.utils.data,
             optimizer: torch.optim, criterion: torch.nn.Module) -> None:
    """ Train the model and save the values in the TrainHistory class

    :param cnn_model: CNN to train
    :param training_class: class to track training history
    :param train_dataloader: Batch of data loaded from the test dataset
    :param optimizer:
    :param criterion:
    :return: None
    """
    cnn_model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(training_class.device), labels.to(training_class.device)
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = correct / total
    training_class.train_losses.append(train_loss)
    training_class.train_accuracies.append(train_accuracy)


def validation(cnn_model: nn.Module, training_class: TrainHistory, test_dataloader: torch.utils.data,
               criterion: torch.nn.Module) -> None:
    """ Evaluate the performance during training using the test dataset and
    save the values in the TrainHistory class

    :param cnn_model: CNN to train
    :param training_class: class to track training history
    :param test_dataloader: Batch of data loaded from the test dataset
    :param criterion:
    :return: None
    """
    cnn_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(training_class.device), labels.to(training_class.device)
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(test_dataloader)
    val_accuracy = correct / total
    training_class.val_losses.append(val_loss)
    training_class.val_accuracies.append(val_accuracy)


def train_model(cnn_model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,
                training_class: TrainHistory) -> None:
    """ Train the CNN, saving the train history to plot it

    :param cnn_model: CNN to train
    :param train_dataloader: Batch of data loaded from the train dataset
    :param test_dataloader: Batch of data loaded from the test dataset
    :param training_class: class to track training history
    :return: None
    """
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss()
    print(f"Using {training_class.device} device")
    cnn_model.to(training_class.device)
    print(f"Starting training with {training_class.epochs} epochs...")
    for epoch in range(training_class.epochs):
        training(cnn_model, training_class, train_dataloader, optimizer, criterion)
        validation(cnn_model, training_class, test_dataloader, criterion)
        print(f'Epoch [{epoch + 1}/{training_class.epochs}], '
              f'Training Loss: {training_class.train_losses[-1]:.4f}, '
              f'Training Accuracy: {training_class.train_accuracies[-1]:.2%}, '
              f'Validation Loss: {training_class.val_losses[-1]:.4f}, '
              f'Validation Accuracy: {training_class.val_accuracies[-1]:.2%}')
        # Save model
        if training_class.val_losses[-1] < training_class.best_loss:
            training_class.best_loss = training_class.val_losses[-1]
            torch.save(cnn_model.state_dict(), training_class.trained_model)
    plot_training_history(training_class)


def training_model(train_dataset_path: str, test_dataset_path: str, image_size: tuple, number_epochs: int,
                   new_model_path: str):
    """Train the CNN, load a batch from the dataset to train the model,
    find classes and number of classes to train the model

    :param train_dataset_path: Directory path of training dataset
    :param test_dataset_path: Directory path of test dataset
    :param image_size: Tuple with values to resize images in dataset
    :param number_epochs: Number of repetitions for the training loop
    :param new_model_path: string with the name of the new model to save
    :return: None
    """
    train_dataset = load_dataset(train_dataset_path, image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = load_dataset(test_dataset_path, image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    num_classes = len(train_dataset.classes)
    cnn_model = CNN(num_classes)
    training_class = TrainHistory(number_epochs, new_model_path)
    train_model(cnn_model, train_dataloader, test_dataloader, training_class)


if __name__ == "__main__":
    training_dataset_path = os.path.abspath('./dataset/Training')
    testing_dataset_path = os.path.abspath('./dataset/Testing')
    target_image_size = (224, 224)
    epochs = 15
    model_path = './brain_tumor_classifier.pth'
    training_model(training_dataset_path, testing_dataset_path, target_image_size, epochs, model_path)
