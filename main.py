from torch import nn
import torch.optim as optim
from train_model import training_model
from evaluate_model import evaluate_trained_model


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 254, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(254)

        self.conv4 = nn.Conv2d(254, 512, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(6 * 6 * 512, 512)
        self.fc2 = nn.Linear(512, 4)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def classifier_custom_cnn():
    """
    :return: None
    """
    target_image_size = (224, 224)
    epochs = 10
    model_path = './brain_tumor_classifier.pth'
    cnn = CNN()
    optimizer = optim.Adam(cnn.parameters(), lr=0.0002)
    loss_fn = nn.CrossEntropyLoss()
    training_model(cnn, target_image_size, epochs, optimizer, loss_fn, model_path)
    evaluate_trained_model(cnn, target_image_size, model_path)


if __name__ == "__main__":
    classifier_custom_cnn()
