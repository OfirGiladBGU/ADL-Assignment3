import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid


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
    def __init__(self, hyperparameters, task_name, valid_size=0):
        # Hyperparameters parse
        self.seed = hyperparameters['seed']
        self.input_size = hyperparameters['input_size']
        self.hidden_size = hyperparameters['hidden_size']
        self.num_classes = hyperparameters['num_classes']
        self.num_epochs = hyperparameters['num_epochs']
        self.batch_size = hyperparameters['batch_size']
        self.learning_rate = hyperparameters['learning_rate']

        # Auxiliaries parse
        self.task_name = task_name
        self.valid_size = valid_size
        if self.valid_size > 0:
            self.valid_mode_active = True
        else:
            self.valid_mode_active = False
        self.model_weights_filename = f'{self.task_name}_{self.seed}_model.pth'

        # Model
        self.model = NeuralNet(self.input_size, self.hidden_size, self.num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Dataloaders parse
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self._prepare_dataloaders()

    def _prepare_dataloaders(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root='./data/',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

        test_dataset = torchvision.datasets.MNIST(root='./data/',
                                                  train=False,
                                                  transform=transforms.ToTensor())

        # Data loader
        if self.valid_size == 0:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)

        else:
            train_dataset, valid_dataset = torch.utils.data.random_split(
                train_dataset,
                [len(test_dataset) - self.valid_size, self.valid_size]
            )

            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)

            self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False)

    def train_model(self):
        train_size = len(self.train_loader.dataset)
        test_size = len(self.test_loader.dataset)
        total_step = len(self.train_loader)

        if self.valid_mode_active:
            valid_size = len(self.valid_loader.dataset)
            min_valid_mean_loss = np.inf
            modes = ['train', 'valid', 'test']
        else:
            valid_size = 0
            min_valid_mean_loss = 0
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

                    # Save model with minimum validation error
                    if min_valid_mean_loss > mean_loss:
                        min_valid_mean_loss = mean_loss
                        torch.save(self.model.state_dict(), self.model_weights_filename)

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

        # Load model with minimum validation error
        if self.valid_mode_active:
            self.model.load_state_dict(torch.load(self.model_weights_filename))

        # Test error after training
        running_loss = 0.0
        misclassified_images = torch.Tensor()
        misclassified_pred_labels = torch.Tensor()
        misclassified_true_labels = torch.Tensor()
        self.model.eval()
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
        print('Final Test Mean Loss Error: {:.4f}'.format(final_test_mean_loss))

        if self.valid_mode_active:
            print('Minimum Valid Mean Loss Error: {:.4f}'.format(min_valid_mean_loss))

        # Plot the Error graph
        plt.xlabel('Epoch')
        plt.ylabel('Dataset Error')
        plt.title(f'{self.task_name}: Train and test error graphs')
        plt.plot(train_error, label='Train Error')
        plt.plot(test_error, label='Test Error')
        if self.valid_mode_active:
            plt.plot(valid_error, label='Valid Error')
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

        if not self.valid_mode_active:
            return final_test_mean_loss
        else:
            return final_test_mean_loss, min_valid_mean_loss


def task1():
    task_name = "Task 1"

    # Hyperparameters
    hyperparameters = {
        "seed": 0,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    }

    # Train model
    trainer = Trainer(hyperparameters=hyperparameters, task_name=task_name)
    final_test_mean_loss = trainer.train_model()
    return final_test_mean_loss


def task2():
    task_name = "Task 2"

    # Hyperparameters
    hyperparameters = {
        "seed": 0,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    }

    seed_list = [0, 20, 40, 60, 80]
    test_errors = list()
    for seed in seed_list:
        hyperparameters["seed"] = seed
        trainer = Trainer(hyperparameters=hyperparameters, task_name=task_name)
        test_error = trainer.train_model()
        test_errors.append(test_error)

    mean_test_error = np.mean(test_errors)
    std_test_error = np.std(test_errors)
    print(
        'Mean Test Error: {:.4f}\n'
        'Standard Deviation of Test Error: {:.4f}'
        .format(mean_test_error, std_test_error)
    )


def task3():
    task_name = "Task 3"
    valid_size = 10000

    # Hyperparameters
    hyperparameters = {
        "seed": 0,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    }

    seed_list = [0, 20, 40, 60, 80]
    test_errors = list()
    valid_errors = list()
    for seed in seed_list:
        hyperparameters["seed"] = seed
        trainer = Trainer(hyperparameters=hyperparameters, task_name=task_name, valid_size=valid_size)
        test_error, valid_error = trainer.train_model()
        test_errors.append(test_error)
        valid_errors.append(valid_error)

    mean_test_error = np.mean(test_errors)
    std_test_error = np.std(test_errors)
    print(
        'Mean Test Error: {:.4f}\n'
        'Standard Deviation of Test Error: {:.4f}'
        .format(mean_test_error, std_test_error)
    )

    mean_valid_error = np.mean(valid_errors)
    std_valid_error = np.std(valid_errors)
    print(
        'Mean Valid Error: {:.4f}\n'
        'Standard Deviation of Valid Error: {:.4f}'
        .format(mean_valid_error, std_valid_error)
    )


if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task1()
    # task2()
    # task3()
