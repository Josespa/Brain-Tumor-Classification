from torch import nn
import torch.optim as optim
from load_data import load_dataset, load_batch, get_classes
from train_model import training_model
from evaluate_model import evaluate_trained_model
from plot import show_images


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

        self.fc1 = nn.Linear(400 * 128, 254)
        self.fc2 = nn.Linear(254, num_classes)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':

    image_size = (224, 224)
    training_dataset = load_dataset('Training', image_size)
    classes = get_classes(training_dataset)
    training_batches = load_batch('Training', training_dataset, batch_size=32)
    show_images(training_batches, classes)
    testing_dataset = load_dataset('Testing', image_size)
    testing_batches = load_batch('Testing', testing_dataset, batch_size=32)

    epochs = 15
    cnn = CNN(num_classes=len(classes))
    optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    model_path = './brain_tumor_classifier.pth'

    training_model(cnn, training_batches, testing_batches, epochs, optimizer, loss_fn, model_path)
    evaluate_trained_model(cnn, testing_batches, classes, model_path)
