import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def task1():
    # Hyperparameters
    input_size = 784
    hidden_size = 500
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Model
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    total_step = len(train_loader)
    modes = ['train', 'test']
    train_error = []
    test_error = []

    for epoch in range(num_epochs):
        for mode in modes:
            running_loss = 0.0

            # Train the model
            if mode == 'train':
                model.train()
                for i, (images, labels) in enumerate(train_loader):
                    images = images.reshape(-1, input_size).to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i + 1) % 100 == 0:
                        print('[Train]: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

                mean_loss = running_loss / train_size
                train_error.append(mean_loss)
                print('[Train]: Epoch [{}/{}], Mean Loss Error: {:.4f}'
                      .format(epoch + 1, num_epochs, mean_loss))

            # Evaluate the model
            else:
                model.eval()
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.reshape(-1, input_size).to(device)
                        labels = labels.to(device)

                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()

                mean_loss = running_loss / test_size
                test_error.append(mean_loss)
                print('[Test]: Epoch [{}/{}], Mean Loss Error: {:.4f}'
                      .format(epoch + 1, num_epochs,  mean_loss))

    # Test error after training
    misclassified_images = torch.Tensor()
    misclassified_pred_labels = torch.Tensor()
    misclassified_true_labels = torch.Tensor()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            original_images = images
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Misclassified images
            misclassified_indices = np.nonzero(predicted != labels)
            misclassified_images = torch.cat((misclassified_images, original_images[misclassified_indices]))
            misclassified_pred_labels = torch.cat((misclassified_pred_labels, predicted[misclassified_indices]))
            misclassified_true_labels = torch.cat((misclassified_true_labels, labels[misclassified_indices]))

        accuracy = 100 * correct / total
        print('Accuracy of the network on the {} test images: {} %'.format(test_size, accuracy))

    # Plot the error graph
    plt.xlabel('Epoch')
    plt.ylabel('Dataset Error')
    plt.title('Task 1: Train and test error graphs')
    plt.plot(train_error, label='Train Error')
    plt.plot(test_error, label='Test Error')
    plt.legend()
    plt.show()

    # Misclassified images
    fig, axs = plt.subplots(2, 5)
    for ii in range(2):
        for jj in range(5):
            idx = 5 * ii + jj
            axs[ii, jj].imshow(misclassified_images[idx].squeeze())
            title = 'Predicted: {}, True: {}'.format(misclassified_pred_labels[idx].item(),
                                                     misclassified_true_labels[idx].item())
            axs[ii, jj].set_title(title)
            axs[ii, jj].axis('off')
    plt.show()


if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task1()
