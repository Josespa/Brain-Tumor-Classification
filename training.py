import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_data import load_dataset
from cnn_model import CNN


class TrainHistory:
    device = str
    best_loss = float
    val_loss = float
    train_losses = list
    val_losses = list
    train_accuracies = list
    val_accuracies = list

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_loss = float('inf')
        self.val_loss = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []


def plot_training_history(result_data):
    """Visualize training history
    :param :
    :return:
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


def find_label_classes(classes: list[str]) -> list[str]:
    """
    :param :
    :return:
    """
    print('classes loaded in train dataset: ' + str(classes))
    return classes


def find_number_of_classes(classes: list[str]) -> int:
    """
    :param :
    :return:
    """
    print('Working with ' + str(len(classes)) + ' classes')
    return len(classes)


def training(cnn_model, train_class, train_dataloader, optimizer, criterion):
    """

    :param cnn_model:
    :param train_class:
    :param train_dataloader:
    :param optimizer:
    :param criterion:
    :return: None
    """
    cnn_model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(train_class.device), labels.to(train_class.device)
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
    train_class.train_losses.append(train_loss)
    train_class.train_accuracies.append(train_accuracy)


def validation(cnn_model, train_class, test_dataloader, criterion):
    """

    :param cnn_model:
    :param train_class:
    :param test_dataloader:
    :param criterion:
    :return: None
    """
    # Validation
    cnn_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(train_class.device), labels.to(train_class.device)
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(test_dataloader)
    val_accuracy = correct / total
    train_class.val_losses.append(val_loss)
    train_class.val_accuracies.append(val_accuracy)


def train_model(cnn_model, train_dataloader, test_dataloader, number_epochs, new_model_path):

    optimizer = optim.Adam(cnn_model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss()
    train_class = TrainHistory()
    print(f"Using {train_class.device} device")
    cnn_model.to(train_class.device)
    # Training loop
    print(f"{number_epochs} epochs ")
    for epoch in range(number_epochs):

        training(cnn_model, train_class, train_dataloader, optimizer, criterion)

        validation(cnn_model, train_class, test_dataloader, criterion)

        print(f'Epoch [{epoch + 1}/{number_epochs}], '
              f'Training Loss: {train_class.train_losses[-1]:.4f}, '
              f'Training Accuracy: {train_class.train_accuracies[-1]:.2%}, '
              f'Validation Loss: {train_class.val_losses[-1]:.4f}, '
              f'Validation Accuracy: {train_class.val_accuracies[-1]:.2%}')
        # Save model
        if train_class.val_losses[-1] < train_class.best_loss:
            train_class.best_loss = train_class.val_losses[-1]
            torch.save(cnn_model.state_dict(), new_model_path)
    plot_training_history(train_class)


def training_model(train_dataset_path: str, test_dataset_path: str, image_size: tuple, number_epochs: int,
                   new_model_path: str):
    """

    :param train_dataset_path:
    :param test_dataset_path:
    :param image_size:
    :param number_epochs:
    :param new_model_path:
    :return:
    """
    train_dataset = load_dataset(train_dataset_path, image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = load_dataset(test_dataset_path, image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    labels_classes = find_label_classes(train_dataset.classes)
    print(f"{labels_classes} classes loaded in train dataset ")
    num_classes = find_number_of_classes(labels_classes)
    cnn_model = CNN(num_classes)
    train_model(cnn_model, train_dataloader, test_dataloader, number_epochs, new_model_path)


if __name__ == "__main__":
    training_dataset_path = os.path.abspath('./dataset/Training')
    testing_dataset_path = os.path.abspath('./dataset/Testing')
    target_image_size = (224, 224)
    epochs = 2
    model_path = './brain_tumor_classifier.pth'
    training_model(training_dataset_path, testing_dataset_path, target_image_size, epochs, model_path)
