import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import Subset
from sklearn.manifold import TSNE
from torch.utils.data import random_split
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    def hidden_layer(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        return out

def train_validate_test( hyperparameters_list, task_name, valid_size=0, shrinkByFactor=1, plotGraph=True, plotImages=True):

        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root='./data/',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        train_dataset = Subset(train_dataset, indices=range(len(train_dataset) // shrinkByFactor))

        test_dataset = torchvision.datasets.MNIST(root='./data/',
                                                  train=False,
                                                  transform=transforms.ToTensor())
        test_dataset = Subset(test_dataset, indices=range(len(test_dataset) // shrinkByFactor))

        num_train = len(train_dataset)
        split_idx = valid_size
        train_set, val_set = random_split(train_dataset, [num_train - split_idx, split_idx])

        all_plots = []
        final_test_errors = []
        best_test_validation_error = (99999999,99999999)
        task4_errors_table = {'Hyperparameters': [], 'Minimal Validation Error': [], 'Test Error': []}
        
        for hyperparameters in hyperparameters_list:
          current_best_test_validation_error = (99999999,99999999)

          seed = hyperparameters['seed']
          input_size = hyperparameters['input_size']
          hidden_size = hyperparameters['hidden_size']
          num_classes = hyperparameters['num_classes']
          num_epochs = hyperparameters['num_epochs']
          batch_size = hyperparameters['batch_size']
          learning_rate = hyperparameters['learning_rate']

          torch.manual_seed(seed)
          # Data loader
          train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
          
          val_loader = torch.utils.data.DataLoader(dataset=val_set,
                              batch_size=batch_size,
                              shuffle=False)

          test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)

          model = NeuralNet(input_size, hidden_size, num_classes).to(device)
          criterion = nn.CrossEntropyLoss()
          optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

          train_errors = []
          valid_errors = []
          test_errors = []

          # train the model
          total_step = len(train_loader)
          for epoch in range(num_epochs):
            total_train_loss = 0
            for i, (images, labels) in enumerate(train_loader):
              images = images.reshape(-1, input_size).to(device)
              labels = labels.to(device)

              outputs = model(images)
              loss = criterion(outputs, labels)

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              total_train_loss += loss.item()

              if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
            
            test_loss, _ = calculate_error(test_loader, model, "test",criterion) # calculate test error every epoch
            train_errors.append(total_train_loss/train_loader.__len__())
            test_errors.append(test_loss/test_loader.__len__())
            if valid_size != 0:
              val_loss, _ = calculate_error(val_loader, model, "validation", criterion)
              valid_errors.append(val_loss/val_loader.__len__())
              if best_test_validation_error[1] > (val_loss/val_loader.__len__()):
                best_test_validation_error = (test_loss/test_loader.__len__(), val_loss/val_loader.__len__())
              if current_best_test_validation_error[1] > (val_loss/val_loader.__len__()):
                current_best_test_validation_error = (test_loss/test_loader.__len__(), val_loss/val_loader.__len__())

          if task_name == "Task 1": # Product for Task 1
            epochs = range(1, num_epochs+1)
            plt.figure()
            plt.plot(epochs, train_errors, label='Training Error')
            plt.plot(epochs, test_errors, label='Test Error')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.title('Training and Test Error')
            plt.legend()
            #plt.show()
            all_plots.append(plt)

          elif task_name == "Task 2": # Product for Task 2
            epochs = range(1, num_epochs+1)
  
            plt.plot(epochs, test_errors, label='Test Error')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.title(f"Seed: {hyperparameters['seed']}") 
            plt.legend()
            #plt.show()
            all_plots.append(plt)
            final_test_errors.append(test_loss/test_loader.__len__())
          
          if valid_size != 0:
            task4_errors_table['Hyperparameters'].append(hyperparameters)
            task4_errors_table['Minimal Validation Error'].append(current_best_test_validation_error[1])
            task4_errors_table['Test Error'].append(current_best_test_validation_error[0])
          
          if task_name == "Task 5":
            x, y = train_dataset[0]

            _ , all_hidden = calculate_error(train_loader, model, "train",criterion, False)
            features = []
            labels = []
            for data, label in train_dataset:  # Iterate over the dataset, ignoring the labels
                features.append(data.numpy())  # Convert the tensor to a NumPy array and append
                labels.append(label)
            features = np.stack(features, axis=0)
            features = features.reshape(1000, 28*28)
            #print(all_hidden.shape)
            tsne = TSNE(n_components=2, random_state=42)  # Set random_state for reproducibility
            embeddings = tsne.fit_transform(features)

            plt.figure()
            plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10')
            plt.colorbar(label='Label')
            plt.title('2D Embedding of X_i')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.show()

            embeddings = tsne.fit_transform(all_hidden)

            plt.figure()
            plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10')
            plt.colorbar(label='Label')
            plt.title('2D Embedding of Z_i')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.show()

        return {
                "plots": all_plots,
                "final_test_errors_mean": np.mean(final_test_errors),
                "final_test_errors_deviation": np.std(final_test_errors),
                "best_test_and_validation_errors": best_test_validation_error,
                "test_and_validation_errors_table": task4_errors_table
                }
          

def calculate_error(data_loader, model, type, criterion, isPrint=True):
  with torch.no_grad():
              correct = 0
              total = 0
              total_loss = 0
              all_hidden = torch.tensor([]).to(device)
              for images, labels in data_loader:
                  images = images.reshape(-1, 28*28).to(device)
                  labels = labels.to(device)
                  outputs = model(images)
                  total_loss += criterion(outputs, labels).item()
                  _ , predicted = torch.max(outputs.data, 1)
                  #all_hidden.append(model.hidden_layer(images).numpy())
                  all_hidden = torch.cat((all_hidden, model.hidden_layer(images)), dim=0)
                  total += labels.size(0)
                  correct += (predicted == labels).sum().item()
              # print(all_hidden.size())
              # print(all_outputs.size())
              if isPrint:
                print('Accuracy of the network on the {} {} images: {} %'.format(total, type, 100 * correct / total))
              return total_loss, all_hidden

def misclasified_examples(data_loader, model, type):
  with torch.no_grad():
              correct = 0
              total = 0
              incorrect_examples = []
              incorrect_predictions = []
              incorrect_labels = []
              for images, labels in data_loader:
                  original_images = images.clone()
                  images = images.reshape(-1, 28*28).to(device)
                  labels = labels.to(device)
                  outputs = model(images)
                  _ , predicted = torch.max(outputs.data, 1)
                  idxs_mask = ((predicted == labels) == False).nonzero()
                  print(idxs_mask)
                  incorrect_examples.append(original_images[idxs_mask].squeeze().cpu().numpy())
                  incorrect_predictions.append(predicted[idxs_mask].squeeze().cpu().numpy())
                  incorrect_labels.append(labels[idxs_mask].squeeze().cpu().numpy())
                  total += labels.size(0)
                  correct += (predicted == labels).sum().item()
              return incorrect_examples, incorrect_predictions, incorrect_labels

def Task1():

    task_name = "Task 1"
    validation_set_size = 0 
    shrinkDatasetByFactor = 10 # instead of using 60k samples, use 60k/shrinkDatasetByFactor

    # Hyperparameters
    hyperparameters = [{
        "seed": 0,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    }]

    answers = train_validate_test(hyperparameters, task_name, validation_set_size, shrinkDatasetByFactor)

    for plot in answers['plots']:
        plot.show()

def Task2():

    task_name = "Task 2"
    valid_size = 0
    shrinkDatasetByFactor = 10

    # Hyperparameters
    hyperparameters = [{
        "seed": 0,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    },
                       {
        "seed": 20,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    },
                       {
        "seed": 40,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    },
                       {
        "seed": 60,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    },
                       {
        "seed": 80,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    }]

    answers = train_validate_test(hyperparameters, task_name, valid_size, shrinkDatasetByFactor)
    for plot in answers['plots']:
        plot.show()
    print(f"Final test errors mean: {answers['final_test_errors_mean']}")
    print(f"Final test errors deviation: {answers['final_test_errors_deviation']}")

def Task3():

    task_name = "Task 3"
    valid_size = 1000
    shrinkDatasetByFactor = 10

    # Hyperparameters
    hyperparameters = [{
        "seed": 0,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    },
                       {
        "seed": 20,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    },
                       {
        "seed": 40,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    },
                       {
        "seed": 60,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    },
                       {
        "seed": 80,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    }]

    answers = train_validate_test(hyperparameters, task_name, valid_size, shrinkDatasetByFactor)
    print(f"Test error that corresponds to minimal validation error: \n Test error: {answers['best_test_and_validation_errors'][0]} Validation error: {answers['best_test_and_validation_errors'][1]}")

def Task4():

    task_name = "Task 4"
    valid_size = 1000
    shrinkDatasetByFactor = 10

    # Hyperparameters
    hyperparameters = [{
        "seed": 0,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    },
                       {
        "seed": 20,
        "input_size": 784,
        "hidden_size": 200,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 200,
        "learning_rate": 0.001
    },
                       {
        "seed": 40,
        "input_size": 784,
        "hidden_size": 700,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 10,
        "learning_rate": 0.0001
    },
                       {
        "seed": 60,
        "input_size": 784,
        "hidden_size": 134,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 1000,
        "learning_rate": 0.005
    },
                       {
        "seed": 80,
        "input_size": 784,
        "hidden_size": 300,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 200,
        "learning_rate": 0.1
    }]

    answers = train_validate_test(hyperparameters, task_name, valid_size, shrinkDatasetByFactor)
    df = pd.DataFrame(answers['test_and_validation_errors_table'])
    display(df)

def Task5():

    task_name = "Task 5"
    valid_size = 0
    shrinkDatasetByFactor = 10

    # Hyperparameters
    hyperparameters = [{
        "seed": 0,
        "input_size": 784,
        "hidden_size": 500,
        "num_classes": 10,
        "num_epochs": 5,
        "batch_size": 100,
        "learning_rate": 0.001
    }]

    answers = train_validate_test(hyperparameters, task_name, valid_size, shrinkDatasetByFactor)

Task1()
Task2()
Task3()
Task4()
Task5()
