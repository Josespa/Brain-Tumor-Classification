from datetime import datetime
import torch
from torch import nn

from plot import show_training_history
from load_data import get_models_directory


class TrainingHistory:
    """ Class to save history during training.

    Class to store information about training history during training including:
    CNN, hyperparameters, loss, accuracy, etc.
    """
    cnn_model = nn.Module
    epochs = int
    optimizer = torch.optim
    loss_fn = torch.nn.Module
    device = str
    min_validation_loss = float
    val_loss = float
    train_losses = list
    val_losses = list
    train_accuracies = list
    val_accuracies = list
    start_time = datetime

    def __init__(self, cnn_model, num_epochs, optimizer, loss_fn):
        self.cnn_model = cnn_model
        self.epochs = num_epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_validation_loss = float('inf')
        self.val_loss = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.start_time = datetime.now()
        print(f"Model to train: \n{self.cnn_model}\nOptimizer: {self.optimizer} \nLoss Function: {self.loss_fn} ")
        print(f"Working device: {self.device}")
        self.cnn_model.to(self.device)


def training(train_history: TrainingHistory, training_batches: torch.utils) -> None:
    """ Train the model and save the values in the TrainingHistory class.

    Loops over the optimization code and the loss and accuracy

    :param train_history: Class to track training history
    :param training_batches: Batch of data loaded from the training dataset
    :return: None
    """
    train_history.cnn_model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(training_batches):
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

    train_loss /= len(training_batches)
    train_history.train_losses.append(train_loss)
    train_accuracy = correct / total
    train_history.train_accuracies.append(train_accuracy)


def validation(train_history: TrainingHistory, testing_batches: torch.utils) -> None:
    """ Evaluates the model’s evaluation against the test data.

    Evaluate the evaluation during training using the test dataset and
    save the values in the TrainingHistory class.

    :param train_history: Class to track training history
    :param testing_batches: Batch of data loaded from the testing dataset
    :return: None
    """
    train_history.cnn_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testing_batches:
            inputs, labels = inputs.to(train_history.device), labels.to(train_history.device)
            outputs = train_history.cnn_model(inputs)
            loss = train_history.loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(testing_batches)
    train_history.val_losses.append(val_loss)
    val_accuracy = correct / total
    train_history.val_accuracies.append(val_accuracy)


def print_train_history(train_history: TrainingHistory, epoch: int) -> None:
    """ Show training history for a given epoch.

    Helps to track training history and monitoring the evaluation during training.

    :param train_history: Class to track training history
    :param epoch: Current iteration in the optimization loop
    :return: None
    """
    print(f"Epoch [{epoch + 1}/{train_history.epochs}], "
          f"Training Loss: {train_history.train_losses[-1]:.4f}, "
          f"Training Accuracy: {train_history.train_accuracies[-1]:.2%}, "
          f"Validation Loss: {train_history.val_losses[-1]:.4f}, "
          f"Validation Accuracy: {train_history.val_accuracies[-1]:.2%}")


def save_best_model(train_history: TrainingHistory, name_model_to_save: str) -> None:
    """ Save the best model in the model folder.

    Save the model with the best evaluation based on validation loss.

    :param train_history: Class to track training history
    :param name_model_to_save: Name of the model to save
    :return: None
    """
    if train_history.val_losses[-1] < train_history.min_validation_loss:
        train_history.min_validation_loss = train_history.val_losses[-1]
        directory_to_save = get_models_directory(name_model_to_save)
        torch.save(train_history.cnn_model.state_dict(), directory_to_save)


def model_to_device(train_history: TrainingHistory) -> None:
    """ Set CNN model to device.

    TrainingHistory class already contains the device, if there is a GPU available set the model into it,
    else set the model to cpu.

    :param train_history: Class to track training history
    :return: None
    """
    print(f"Working device: {train_history.device}")
    train_history.cnn_model.to(train_history.device)


def training_model(cnn_model: nn.Module, training_batches: torch.utils, testing_batches: torch.utils,
                   epochs: int, optimizer: torch.optim, loss_fn: nn.Module, name_to_save_model: str) -> None:
    """ Train and optimize the model with an optimization loop.

    Use the hyperparameters, to train and optimize the CNN model with an optimization loop.
    Each iteration of the optimization loop corresponds to one epoch.

    :param cnn_model: CNN model to be trained
    :param training_batches: Batch of data loaded from the train dataset
    :param testing_batches: Batch of data loaded from the test dataset
    :param epochs: Number of repetitions for the training loop
    :param optimizer: Algorithm to adjust the model parameters and reduce model error in each training step
    :param loss_fn: Loss function to minimize during training
    :param name_to_save_model: String with the name of the new model to save
    :return: None
    """
    train_history = TrainingHistory(cnn_model, epochs, optimizer, loss_fn)
    print("Starting training...")
    for epoch in range(epochs):
        training(train_history, training_batches)
        validation(train_history, testing_batches)
        print_train_history(train_history, epoch)
        save_best_model(train_history, name_to_save_model)
    print("Finishing training.")
    print(f"Running time of the script: {datetime.now() - train_history.start_time}")
    show_training_history(train_history)
