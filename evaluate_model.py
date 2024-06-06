import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from load_data import load_dataset, load_batch
from plot import show_confusion_matrix


class PerformanceHistory:
    """ Save information to evaluate the trained model performance
    """
    cnn_model = nn.Module
    device = str
    labels_classes = list
    correct_prediction = dict
    total_prediction = dict
    labels_true = list
    labels_predicted = list
    conf_matrix = np.array
    start_time = datetime

    def __init__(self, cnn_model, classes):
        self.cnn_model = cnn_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_model.to(self.device)
        self.labels_classes = classes
        self.correct_prediction = {class_name: 0 for class_name in classes}
        self.total_prediction = {class_name: 0 for class_name in classes}
        self.labels_true = []
        self.labels_predicted = []
        self.start_time = datetime.now()
        print(f"Evaluating model: \n{self.cnn_model}")


def accuracy_for_class(test_dataloader: torch.utils.data,
                       performance: PerformanceHistory) -> None:
    """ Get accuracy for each class and print it,
    generate array for classification_report
    :param test_dataloader: Batch of data loaded from the test dataset
    :param performance: class to track evaluation performance
    :return: None
    """
    with torch.no_grad():
        for (images, labels) in test_dataloader:
            images = images.to(performance.device)
            outputs = performance.cnn_model(images)
            _, predictions = torch.max(outputs, 1)
            performance.labels_true = np.append(performance.labels_true, labels)
            performance.labels_predicted = np.append(performance.labels_predicted, predictions.detach().cpu().numpy())


def evaluate_trained_model(cnn_model: nn.Module, image_size: tuple, model_path: str) -> None:
    """

    :param cnn_model:
    :param image_size:
    :param model_path:
    :return:
    """
    test_dataset = load_dataset("Testing", image_size)
    performance = PerformanceHistory(cnn_model, test_dataset.classes)
    test_dataloader = load_batch("Testing", test_dataset, 64)
    performance.cnn_model.load_state_dict(torch.load(model_path))
    performance.cnn_model.eval()
    print(f"Evaluating model...{model_path} \n")
    print(performance.cnn_model, "\n")
    accuracy_for_class(test_dataloader, performance)
    print(classification_report(performance.labels_true, performance.labels_predicted))
    performance.conf_matrix = confusion_matrix(performance.labels_true, performance.labels_predicted)
    show_confusion_matrix(performance)
    print(f"\n Running time of the script: {datetime.now() - performance.start_time}")
