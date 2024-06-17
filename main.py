import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE


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


class Trainer:
    def __init__(self, hyperparameters, dataloaders):
        # Hyperparameters parse
        self.input_size = hyperparameters['input_size']
        self.hidden_size = hyperparameters['hidden_size']
        self.num_classes = hyperparameters['num_classes']
        self.num_epochs = hyperparameters['num_epochs']
        self.batch_size = hyperparameters['batch_size']
        self.learning_rate = hyperparameters['learning_rate']

        # Model
        self.model = NeuralNet(self.input_size, self.hidden_size, self.num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Dataloaders parse
        self.train_loader = dataloaders.get('train', None)
        self.valid_loader = dataloaders.get('valid', None)
        self.test_loader = dataloaders.get('test', None)

        if self.valid_loader is not None:
            self.valid_mode_active = True
        else:
            self.valid_mode_active = False

    def train_model(self):
        train_size = len(self.train_loader.dataset)
        test_size = len(self.test_loader.dataset)
        total_step = len(self.train_loader)

        if self.valid_mode_active:
            valid_size = len(self.valid_loader.dataset)
            modes = ['train', 'valid', 'test']
        else:
            valid_size = 0
            modes = ['train', 'test']

        train_error = list()
        valid_error = list()
        test_error = list()

        for epoch in range(self.num_epochs):
            for mode in modes:
                running_loss = 0.0

                # Train mode
                if mode == 'train':
                    self.model.train()
                    for i, (images, labels) in enumerate(self.train_loader):
                        images = images.reshape(-1, self.input_size).to(device)
                        labels = labels.to(device)

                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        running_loss += loss.item()

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        if (i + 1) % 100 == 0:
                            print('[Train]: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                                  .format(epoch + 1, self.num_epochs, i + 1, total_step, loss.item()))

                    mean_loss = running_loss / train_size
                    train_error.append(mean_loss)
                    print('[Train]: Epoch [{}/{}], Mean Loss Error: {:.4f}'
                          .format(epoch + 1, self.num_epochs, mean_loss))

                # Validation mode
                elif mode == "valid":
                    self.model.eval()
                    with torch.no_grad():
                        for images, labels in self.valid_loader:
                            images = images.reshape(-1, self.input_size).to(device)
                            labels = labels.to(device)

                            outputs = self.model(images)
                            loss = self.criterion(outputs, labels)
                            running_loss += loss.item()

                    mean_loss = running_loss / valid_size
                    valid_error.append(mean_loss)
                    print('[Valid]: Epoch [{}/{}], Mean Loss Error: {:.4f}'
                          .format(epoch + 1, self.num_epochs, mean_loss))
                # Evaluate mode
                else:
                    self.model.eval()
                    with torch.no_grad():
                        for images, labels in self.test_loader:
                            images = images.reshape(-1, self.input_size).to(device)
                            labels = labels.to(device)

                            outputs = self.model(images)
                            loss = self.criterion(outputs, labels)
                            running_loss += loss.item()

                    mean_loss = running_loss / test_size
                    test_error.append(mean_loss)
                    print('[Test]: Epoch [{}/{}], Mean Loss Error: {:.4f}'
                          .format(epoch + 1, self.num_epochs, mean_loss))

        # Test error after training
        running_loss = 0.0
        misclassified_images = torch.Tensor()
        misclassified_pred_labels = torch.Tensor()
        misclassified_true_labels = torch.Tensor()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                original_images = images
                images = images.reshape(-1, self.input_size).to(device)
                labels = labels.to(device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

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

        final_test_mean_loss = running_loss / test_size
        print('Final Test Mean Loss Error: {:.4f}'.format(self.num_epochs, final_test_mean_loss))

        # Plot the Error graph
        plt.xlabel('Epoch')
        plt.ylabel('Dataset Error')
        plt.title('Task 1: Train and test error graphs')
        plt.plot(train_error, label='Train Error')
        plt.plot(test_error, label='Test Error')
        plt.legend()
        plt.show()

        # Plot the Misclassified images
        fig, axs = plt.subplots(2, 5)
        for ii in range(2):
            for jj in range(5):
                idx = 5 * ii + jj
                axs[ii, jj].imshow(misclassified_images[idx].squeeze())
                title = (
                    'Pred: {}\n'
                    'True: {}'
                    .format(misclassified_pred_labels[idx].item(), misclassified_true_labels[idx].item())
                )
                axs[ii, jj].set_title(title)
                axs[ii, jj].axis('off')
        plt.show()

        return final_test_mean_loss


def task1(manual_seed=0):
    torch.manual_seed(manual_seed)

    # Hyperparameters
    hyperparameters = {
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    }

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
                                               batch_size=hyperparameters["batch_size"],
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=hyperparameters["batch_size"],
                                              shuffle=False)

    dataloaders = {
        "train": train_loader,
        "test": test_loader
    }

    # Model
    trainer = Trainer(hyperparameters=hyperparameters, dataloaders=dataloaders)
    final_test_mean_loss = trainer.train_model()
    return final_test_mean_loss


def task2():
    manual_seed_list = random.sample(population=range(0, 100), k=5)
    test_errors = list()
    for manual_seed in manual_seed_list:
        test_error = task1(manual_seed=manual_seed)
        test_errors.append(test_error)

    mean_test_error = np.mean(test_errors)
    std_test_error = np.std(test_errors)
    print(
        'Mean Test Error: {:.4f}\n'
        'Standard Deviation of Test Error: {:.4f}'
        .format(mean_test_error, std_test_error)
    )


def task3():
    pass


if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task1()
    # task2()
