import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from train_model import get_models_directory
from plot import show_confusion_matrix


class EvaluationResults:
    """ Class to save the results during evaluation of the trained model.

    """
    cnn_model = nn.Module
    device = str
    classes = list
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
        self.classes = classes
        self.correct_prediction = {class_name: 0 for class_name in classes}
        self.total_prediction = {class_name: 0 for class_name in classes}
        self.labels_true = []
        self.labels_predicted = []
        self.start_time = datetime.now()
        print(f"Evaluating model: \n{self.cnn_model}")


def evaluate_accuracy_per_class(testing_batches: torch.utils.data, evaluation: EvaluationResults) -> None:
    """ Get accuracy for each class.

    Generates array for classification_report.

    :param testing_batches: Batch of data loaded from the test dataset
    :param evaluation: class to track evaluation
    :return: None
    """
    with torch.no_grad():
        for (images, labels) in testing_batches:
            images = images.to(evaluation.device)
            outputs = evaluation.cnn_model(images)
            _, predictions = torch.max(outputs, 1)
            evaluation.labels_true = np.append(evaluation.labels_true, labels)
            evaluation.labels_predicted = np.append(evaluation.labels_predicted, predictions.detach().cpu().numpy())


def evaluate_trained_model(cnn_model: nn.Module, testing_batches: torch.utils, classes: list[str],
                           model_path: str) -> None:
    """ Evaluate a trained model for classification.

    Generates confusion matrix and classification report for each class.

    :param cnn_model: CNN model to be evaluated
    :param testing_batches: Batch of data loaded from the test dataset
    :param classes: Classes in the dataset.
    :param model_path: String with the name of the model to evaluate
    :return: None
    """
    print(f"Evaluating model...{model_path}")
    evaluation = EvaluationResults(cnn_model, classes)
    model_path = get_models_directory(model_path)
    evaluation.cnn_model.load_state_dict(torch.load(model_path))
    evaluation.cnn_model.eval()
    evaluate_accuracy_per_class(testing_batches, evaluation)
    print(classification_report(evaluation.labels_true, evaluation.labels_predicted))
    result_cm = confusion_matrix(evaluation.labels_true, evaluation.labels_predicted)
    result_cm_display = ConfusionMatrixDisplay(confusion_matrix=result_cm, display_labels=classes)
    show_confusion_matrix(result_cm_display)
    print(f"\n Running time of the script: {datetime.now() - evaluation.start_time}")
