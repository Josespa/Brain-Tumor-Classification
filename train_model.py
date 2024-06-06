from datetime import datetime
import torch
from torch import nn
import torch.optim as optim
from load_data import load_dataset, load_batch
from plot import show_images, plot_training_history


class TrainHistory:
    """ Save information during training
    """
    cnn_model = nn.Module
    epochs = int
    optimizer = torch.optim
    loss_fn = torch.nn.Module
    device = str
    best_loss = float
    val_loss = float
    train_losses = list
    val_losses = list
    train_accuracies = list
    val_accuracies = list
    start_time = datetime

    def __init__(self, cnn_model, num_epochs, optimization_algorithm, loss_function):
        self.cnn_model = cnn_model
        self.epochs = num_epochs
        self.optimizer = optimization_algorithm
        self.loss_fn = loss_function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_loss = float('inf')
        self.val_loss = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.start_time = datetime.now()
        print(f"Model to train: \n{self.cnn_model}\nOptimizer: {self.optimizer} \nLoss Function: {self.loss_fn} ")


def training(train_history: TrainHistory, train_dataloader: torch.utils) -> None:
    """ Train the model and save the values in the TrainHistory class

    :param train_history: class to track training history
    :param train_dataloader: Batch of data loaded from the test dataset
    :return: None
    """
    train_history.cnn_model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(train_history.device), labels.to(train_history.device)
        train_history.optimizer.zero_grad()
        outputs = train_history.cnn_model(inputs)
        loss = train_history.loss_fn(outputs, labels)
        loss.backward()
        train_history.optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = correct / total
    train_history.train_losses.append(train_loss)
    train_history.train_accuracies.append(train_accuracy)


def validation(train_history: TrainHistory, test_dataloader: torch.utils) -> None:
    """ Evaluate the performance during training using the test dataset and
    save the values in the TrainHistory class

    :param train_history: class to track training history
    :param test_dataloader: Batch of data loaded from the test dataset
    :return: None
    """
    train_history.cnn_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(train_history.device), labels.to(train_history.device)
            outputs = train_history.cnn_model(inputs)
            loss = train_history.loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(test_dataloader)
    val_accuracy = correct / total
    train_history.val_losses.append(val_loss)
    train_history.val_accuracies.append(val_accuracy)


def print_train_history(train_history, epoch):
    """

    :param train_history:
    :param epoch:
    :return:
    """
    print(f"Epoch [{epoch + 1}/{train_history.epochs}], "
          f"Training Loss: {train_history.train_losses[-1]:.4f}, "
          f"Training Accuracy: {train_history.train_accuracies[-1]:.2%}, "
          f"Validation Loss: {train_history.val_losses[-1]:.4f}, "
          f"Validation Accuracy: {train_history.val_accuracies[-1]:.2%}")


def save_best_model(train_history, name_model_to_save):
    """

    :param train_history:
    :param name_model_to_save:
    :return:
    """
    if train_history.val_losses[-1] < train_history.best_loss:
        train_history.best_loss = train_history.val_losses[-1]
        torch.save(train_history.cnn_model.state_dict(), name_model_to_save)


def model_to_device(train_history):
    """
    :param train_history:
    :return:
    """
    print(f"Working device: {train_history.device}")
    train_history.cnn_model.to(train_history.device)


def training_model(cnn_model: nn.Module, image_size: tuple, number_epochs: int, optimization_algorithm: torch.optim,
                   loss_function: nn.Module, name_name_trained_model: str):
    """
    Train the CNN, load a batch from the dataset to train the model
    :param cnn_model:
    :param image_size: Tuple with values to resize images in dataset
    :param number_epochs: Number of repetitions for the training loop
    :param optimization_algorithm:
    :param loss_function:
    :param name_name_trained_model: string with the name of the new model to save
    :return:
    """
    train_history = TrainHistory(cnn_model, number_epochs, optimization_algorithm, loss_function)
    train_dataset = load_dataset("Training", image_size)
    train_dataloader = load_batch("Training", train_dataset, 64)
    test_dataset = load_dataset("Testing", image_size)
    test_dataloader = load_batch("Testing", test_dataset, 64)
    model_to_device(train_history)
    for epoch in range(train_history.epochs):
        training(train_history, train_dataloader)
        validation(train_history, test_dataloader)
        print_train_history(train_history, epoch)
        save_best_model(train_history, name_name_trained_model)
    print("Finish training")
    print(f"Running time of the script: {datetime.now() - train_history.start_time}")
    plot_training_history(train_history)
