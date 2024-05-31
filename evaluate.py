import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from cnn_model import CNN
from load_data import load_dataset


class EvaluatePerformance:
    """ Save information to evaluate the trained model performance
    """
    device = str
    labels_classes = list
    correct_prediction = dict
    total_prediction = dict
    labels_true = list
    labels_predicted = list
    conf_matrix = np.array

    def __init__(self, classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels_classes = classes
        self.correct_prediction = {class_name: 0 for class_name in classes}
        self.total_prediction = {class_name: 0 for class_name in classes}
        self.labels_true = []
        self.labels_predicted = []


def accuracy_for_class(cnn_model: nn.Module, test_dataloader: torch.utils.data,
                       evaluating_class: EvaluatePerformance) -> None:
    """ Get accuracy for each class and print it,
    generate array for classification_report
    :param cnn_model: CNN with trained model
    :param test_dataloader: Batch of data loaded from the test dataset
    :param evaluating_class: class to track evaluation performance
    :return: None
    """
    with torch.no_grad():
        for (images, labels) in test_dataloader:
            images, labels = images.to(evaluating_class.device), labels.to(evaluating_class.device)
            outputs = cnn_model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    evaluating_class.correct_prediction[evaluating_class.labels_classes[label]] += 1
                evaluating_class.total_prediction[evaluating_class.labels_classes[label]] += 1
            # arrays for classification_report
            evaluating_class.labels_true = np.append(evaluating_class.labels_true, labels.detach().cpu().numpy())
            evaluating_class.labels_predicted = np.append(evaluating_class.labels_predicted,
                                                          predictions.detach().cpu().numpy())
    for class_name, correct_count in evaluating_class.correct_prediction.items():
        accuracy = 100 * float(correct_count) / evaluating_class.total_prediction[class_name]
        print(f'Accuracy for class: {class_name:5s} is {accuracy:.1f} %')


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


def evaluate_trained_model(dataset_path: str, image_size: tuple, model_path: str) -> None:
    """ Evaluate trained model performance
    :param dataset_path: Directory path of test dataset
    :param image_size: Tuple with values to resize images in dataset
    :param model_path: string with the trained model name to evaluate
    :return: None
    """
    test_dataset = load_dataset(dataset_path, image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluating_class = EvaluatePerformance(test_dataset.classes)
    num_classes = len(evaluating_class.labels_classes)
    cnn_model = CNN(num_classes)
    cnn_model.to(evaluating_class.device)
    # loading trained model
    cnn_model.load_state_dict(torch.load(model_path))
    cnn_model.eval()
    print(f"Evaluating model...{model_path}")
    print(cnn_model)
    accuracy_for_class(cnn_model, test_dataloader, evaluating_class)
    print(classification_report(evaluating_class.labels_true, evaluating_class.labels_predicted), '\n\n')
    evaluating_class.conf_matrix = confusion_matrix(evaluating_class.labels_true, evaluating_class.labels_predicted)
    show_confusion_matrix(evaluating_class)


if __name__ == '__main__':
    test_dataset_path = os.path.abspath('./dataset/Testing')
    target_image_size = (224, 224)
    trained_model_path = './brain_tumor_classifier.pth'
    evaluate_trained_model(test_dataset_path, target_image_size, trained_model_path)
