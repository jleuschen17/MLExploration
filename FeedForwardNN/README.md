# Homework IV: Feed Foreward Neural Networks
#### Author: Joe Leuschen
## Abstract
This project explores the implementation of a three-layer feed-forward neural network on two datasets: a custom dataset of 31 data points and the popular MNIST digit dataset. For the custom dataset, different training and testing splits are experimented with, and the resulting least-square errors are compared betIen the neural network and previously used regression models, including linear, parabolic, and polynomial functions. For the MNIST dataset, the first 20 PCA modes of the digit images are computed and used to build a feed-forward neural network. The model is trained and its performance is compared with other classifiers, namely LSTM, SVM (support vector machines), and decision trees. The complete analysis provides valuable insights into the applicability and efficacy of neural networks for different types of data and allows for an understanding of the strengths and Iaknesses of different modeling approaches.
## Introduction
In this project, I investigate the applicability and efficacy of three-layer feed-forward neural networks in modeling and classifying data from two different datasets: a custom dataset consisting of 31 real-valued data points and the widely used MNIST dataset, which contains images of handwritten digits. Various aspects of neural networks, such as training, testing, and comparison with other methods, are explored in detail, providing a comprehensive understanding of their advantages and limitations compared to conventional regression models and classification techniques.

Initially, the custom dataset is fitted using a three-layer feed-forward neural network, folloId by an analysis of the least-square errors across different training and testing data splits. This is done to better understand the neural network's performance compared to linear, parabolic, and polynomial functions previously applied to this dataset. Thereafter, the focus is shifted to the MNIST dataset, where the first 20 PCA modes of the digit images are computed and used as input features for the neural network-based classifier.

Following the implementation of a feed-forward neural network for the MNIST dataset, its performance is evaluated and compared against other popular classification techniques, including Long-Short Term Memory Networks (LSTM), Support Vector Machines (SVM), and Decision Trees. The results obtained from this analysis highlight the strengths and Iaknesses of neural networks as a modeling and classification tool, offering valuable insights into their suitability for various types of data and problem-solving scenarios.

Throughout the course of this project, various important concepts and methods are employed, such as data preprocessing, Singular Value Decomposition (SVD), PCA, and a wide array of machine learning techniques. As a result, a deeper understanding of the subject matter is attained, particularly concerning the use of neural networks for data modeling and classification tasks.

## Theoretical Background

A feed-forward neural network is a type of artificial neural network in which information flows from input to output through various interconnected layers, without any loops or cycles. Three-layer feed-forward neural networks consist of an input layer, hidden layer, and an output layer. This configuration is useful in capturing the non-linear relationships within the data, allowing for much more complex and accurate models compared to linear regression techniques.

For the supervised learning aspect of the project, the learning algorithms rely on labeled data for training and optimization purposes. In this context, an algorithm in supervised learning learns to map input data to output labels by minimizing a predefined loss function. Specifically, neural networks utilized for classification tasks are optimized using an error metric, like cross-entropy loss, to compare predicted labels with actual labels. Throughout the learning process, the Iights and biases within the network layers are adjusted to minimize the loss function, which helps the model to generalize to new, unseen data.

Principal Component Analysis (PCA) refers to a linear dimensionality reduction technique that is widely used in data analysis, visualization, and preprocessing. By computing the first few principal components of the dataset, one can capture the significant majority of the variation within the dataset while significantly reducing the input feature dimensions. In this project, PCA is employed as a preprocessing step on the MNIST dataset, using the first 20 principal components as input features for the neural network classifier. The technique aids in reducing computational complexity and helps the algorithm converge faster during training.

As part of the project, comparative analysis of the neural network-based classification methods with other popular techniques, such as Support Vector Machines (SVM), Long-Short Term Memory Networks (LSTM), and Decision Trees, is conducted. Each of these approaches possesses its advantages and limitations. For instance, SVMs are suitable for high-dimensional spaces and small sample sizes, while Decision Trees are highly interpretable and robust against outliers. On the other hand, LSTMs are a type of recurrent neural network that can capture long-range dependencies and make predictions in sequences, which is highly practical for time-series data.

In conclusion, this project brings together essential machine learning concepts, including feed-forward neural networks, supervised learning, PCA, and various classification techniques to analyze and model complex data. By thoroughly understanding the strengths and limitations of these methods and applying them effectively, valuable insights can be drawn from the data, driving better decision-making and problem-solving in real-world scenarios.


## Algorithm Analysis

### Problem I: Fitting a Neural Network Model

1. Import the necessary libraries and define the dataset.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
```

Load the data and create tensors for each point in X and Y.

```python
X = torch.tensor(np.arange(0, 31), dtype=torch.float32).view(-1, 1)
Y = torch.tensor(np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53]), dtype=torch.float32).view(-1, 1)
```

2. Define the neural network model class.

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
```

3. Split the training and test datasets depending on the case, train the neural network model with the specified number of epochs, and compute the least-square error.

```python
def train_and_test_nn(train_indices, test_indices, epochs):
    model = NeuralNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    train_X = X[train_indices]
    train_Y = Y[train_indices]
    test_X = X[test_indices]
    test_Y = Y[test_indices]

    num_epochs = epochs
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_X)
        loss = criterion(output, train_Y)
        loss.backward()
        optimizer.step()
        if (epoch % 100) == 0:
            print(f"Epoch {epoch} loss: {loss}")

    preds_train = model(train_X)
    train_loss = criterion(preds_train, train_Y)

    preds_test = model(test_X)
    test_loss = criterion(preds_test, test_Y)
    return train_loss, test_loss
```

4. Train and test the models for both cases (different train test splits) and print the results.

```python
print("Case (ii): First 20 points as training data")
train_error, test_error = train_and_test_nn(np.arange(20), np.arange(20, 30), 2000)
print(f"\nTrain Error: {train_error}\nTest Error: {test_error}")

print("\nCase (iii): First 10 and last 10 points as training data")
train_error2, test_error2 = train_and_test_nn(np.hstack([np.arange(10), np.arange(21, 31)]), np.arange(10, 21), 2000)
print(f"\nTrain Error: {train_error2}\nTest Error: {test_error2}")
```

Neural Network Results: 
```
Case (ii): First 20 points as training data

Train Error: 3.8399765491485596
Test Error: 5.16021203994751

Case (iii): First 10 and last 10 points as training data

Train Error: 2.8878626823425293
Test Error: 7.3747358322143555
```

#### Comparison to Polynomial Regression Models
```python
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
# Case I
X_train = X[:20]
X_test = X[20:]
Y_train = Y[:20]
Y_test = Y[20:]
def calc_error(coeffs, X, Y):
    function = np.poly1d(coeffs)
    pred = function(X)
    errors2 = (pred - Y) ** 2
    return math.sqrt(np.sum(errors2) / len(X))
line = np.polyfit(X_train, Y_train, 1)
parabola = np.polyfit(X_train, Y_train, 2)
poly = np.polyfit(X_train, Y_train, 19)

error_line_train = calc_error(line, X_train, Y_train)
error_parabola_train = calc_error(parabola, X_train, Y_train)
error_poly_train = calc_error(poly, X_train, Y_train)
error_line_test = calc_error(line, X_test, Y_test)
error_parabola_test = calc_error(parabola, X_test, Y_test)
error_poly_test = calc_error(poly, X_test, Y_test)


# Case II
X_train2 = X[:10]
X_train2 = np.concatenate((X_train2, X[21:]))
Y_train2 = Y[:10]
Y_train2 = np.concatenate((Y_train2, Y[21:]))
X_test2 = X[10:21]
Y_test2 = Y[10:21]
line2 = np.polyfit(X_train2, Y_train2, 1)
parabola2 = np.polyfit(X_train2, Y_train2, 2)
poly2 = np.polyfit(X_train2, Y_train2, 19)
error_line_train2 = calc_error(line2, X_train2, Y_train2)
error_parabola_train2 = calc_error(parabola2, X_train2, Y_train2)
error_poly_train2 = calc_error(poly2, X_train2, Y_train2)
error_line_test2 = calc_error(line2, X_test2, Y_test2)
error_parabola_test2 = calc_error(parabola2, X_test2, Y_test2)
error_poly_test2 = calc_error(poly2, X_test2, Y_test2)
```
Polynomial Regression ResultS:
```
Case (ii): First 20 points as training data

Error for Line (Train): 2.242749386808538
Error for Parabola (Train): 2.1255393482773766
Error for Polynomial (Train): 0.028351503968806435
Error for Line (Test): 3.36363873604787
Error for Parabola (Test): 8.713651781874919
Error for Polynomial (Test): 28617752784.428474

Case (iii): First 10 and last 10 points as training data

Error for Line (Train): 1.851669904329375
Error for Parabola (Train): 1.8508364115957907
Error for Polynomial (Train): 0.1638133765080727
Error for Line (Test): 2.8065076975181618
Error for Parabola (Test): 2.774982896893291
Error for Polynomial (Test): 483.9099124568562
```


### Problem II: Comparison of Classifiers on the MNIST Dataset

1. Load the MNIST dataset and preprocess it.

```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST('~/.torch/datasets', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('~/.torch/datasets', train=False, download=True, transform=transform)
```

2. Compute the first 20 PCA modes of the digit images.

```python
train_data = train_dataset.data.numpy().reshape(-1, 28*28)
pca = PCA(n_components=20)
pca.fit(train_data)
print(f'First 20 PCA Modes:\n{pca.components_}')
```

3. Train a feed-forward neural network to classify the digits

Create data loaders
```python
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
```

Define the feed-forward neural network model class.

```python
class FeedForwardNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

Initialize the model, loss function, and optimizer.

```python
    model = FeedForwardNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Train the neural network

```python
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        predictions = model(data)

        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
```

Compute test accuracy for the model
```python
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        data = data.view(data.size(0), -1)
        predictions = model(data)
        _, predicted = torch.max(predictions.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}')
```
Results:

```
Accuracy of the network on the 10000 test images: 96.49
```

4. Prepping data for SVM, LSTM, and Decision Tree Classifiers

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).T
X_test = X_test.reshape(10000, 784).T
```

5. Decision tree classifier and SVM algorithms are implemented and trained on the MNIST dataset.

```python
# Tree
tree = DecisionTreeClassifier(random_state=44)
tree.fit(X_train.T, y_train)
y_pred_tree = tree.predict(X_test.T)
accuracy = accuracy_score(y_test, y_pred_tree)
print(accuracy)

#SVM
svm = SVC(probability=False)
svm.fit(X_train.T, y_train)
y_pred_svm = svm.predict(X_test.T)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(accuracy_svm)
```

Accuracy Scores: 
```
Decision Tree: 0.8769
Support Vector Machine: 0.9792
```

6. Create, train, and test LSTM model

```python

class MnistLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MnistLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out

input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MnistLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28, 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader),
                                                                     loss.item()))

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28, 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {:.2f}%'.format(100 * correct / total))
```
```
Accuracy of the model on the test images: 98.94%
```

## Computational Results

### Problem I: Fitting a Neural Network Model


#### Three Layer Feed Forward Neural Network: 
```
Case (ii): First 20 points as training data
Epoch 0 loss: 1408.250244140625
Epoch 100 loss: 5.858303070068359
Epoch 200 loss: 5.080464839935303
Epoch 300 loss: 4.916697978973389
Epoch 400 loss: 4.752251625061035
Epoch 500 loss: 4.680600643157959
Epoch 600 loss: 4.656827926635742
Epoch 700 loss: 4.641770362854004
Epoch 800 loss: 4.622031211853027
Epoch 900 loss: 4.598869323730469
Epoch 1000 loss: 4.574982166290283
Epoch 1100 loss: 4.553575038909912
Epoch 1200 loss: 4.53945779800415
Epoch 1300 loss: 4.535706043243408
Epoch 1400 loss: 4.54232931137085
Epoch 1500 loss: 4.538817405700684
Epoch 1600 loss: 4.669854640960693
Epoch 1700 loss: 4.6919403076171875
Epoch 1800 loss: 4.530725955963135
Epoch 1900 loss: 4.530873775482178

Train Error: 4.737574577331543
Test Error: 6.215279579162598

Case (iii): First 10 and last 10 points as training data
Epoch 0 loss: 2047.760986328125
Epoch 100 loss: 3.8218345642089844
Epoch 200 loss: 3.124260425567627
Epoch 300 loss: 2.7281723022460938
Epoch 400 loss: 2.4249110221862793
Epoch 500 loss: 2.227436065673828
Epoch 600 loss: 2.1628828048706055
Epoch 700 loss: 2.1287589073181152
Epoch 800 loss: 2.0944018363952637
Epoch 900 loss: 2.056519031524658
Epoch 1000 loss: 2.0614352226257324
Epoch 1100 loss: 1.9709501266479492
Epoch 1200 loss: 1.9450737237930298
Epoch 1300 loss: 1.9940156936645508
Epoch 1400 loss: 1.9012187719345093
Epoch 1500 loss: 1.8679075241088867
Epoch 1600 loss: 1.9033313989639282
Epoch 1700 loss: 1.8417894840240479
Epoch 1800 loss: 1.9970893859863281
Epoch 1900 loss: 1.8381969928741455

Train Error: 1.8540862798690796
Test Error: 7.464297294616699
```
#### Linear, Quadratic, and 19th-degree Polynomial Regression Fits:
```
Case (ii): First 20 points as training data

Error for Line (Train): 2.242749386808538
Error for Parabola (Train): 2.1255393482773766
Error for Polynomial (Train): 0.028351503968806435
Error for Line (Test): 3.36363873604787
Error for Parabola (Test): 8.713651781874919
Error for Polynomial (Test): 28617752784.428474

Case (iii): First 10 and last 10 points as training data

Error for Line (Train): 1.851669904329375
Error for Parabola (Train): 1.8508364115957907
Error for Polynomial (Train): 0.1638133765080727
Error for Line (Test): 2.8065076975181618
Error for Parabola (Test): 2.774982896893291
Error for Polynomial (Test): 483.9099124568562
```

### Problem II: Comparison of Classifiers on the MNIST Dataset

#### MNIST Feed Forward Network
```
Epoch [1/10], Step [100/938], Loss: 1.7901
Epoch [1/10], Step [200/938], Loss: 0.8795
Epoch [1/10], Step [300/938], Loss: 0.6356
Epoch [1/10], Step [400/938], Loss: 0.5233
Epoch [1/10], Step [500/938], Loss: 0.5418
Epoch [1/10], Step [600/938], Loss: 0.3958
Epoch [1/10], Step [700/938], Loss: 0.3529
Epoch [1/10], Step [800/938], Loss: 0.4990
Epoch [1/10], Step [900/938], Loss: 0.4642
Epoch [2/10], Step [100/938], Loss: 0.3069
Epoch [2/10], Step [200/938], Loss: 0.3414
Epoch [2/10], Step [300/938], Loss: 0.3336
Epoch [2/10], Step [400/938], Loss: 0.4779
Epoch [2/10], Step [500/938], Loss: 0.4031
Epoch [2/10], Step [600/938], Loss: 0.3548
Epoch [2/10], Step [700/938], Loss: 0.2270
Epoch [2/10], Step [800/938], Loss: 0.1556
Epoch [2/10], Step [900/938], Loss: 0.3027
Epoch [3/10], Step [100/938], Loss: 0.2148
Epoch [3/10], Step [200/938], Loss: 0.4010
Epoch [3/10], Step [300/938], Loss: 0.1748
Epoch [3/10], Step [400/938], Loss: 0.2493
Epoch [3/10], Step [500/938], Loss: 0.3040
Epoch [3/10], Step [600/938], Loss: 0.3756
Epoch [3/10], Step [700/938], Loss: 0.2095
Epoch [3/10], Step [800/938], Loss: 0.4183
Epoch [3/10], Step [900/938], Loss: 0.1744
Epoch [4/10], Step [100/938], Loss: 0.2680
Epoch [4/10], Step [200/938], Loss: 0.1909
Epoch [4/10], Step [300/938], Loss: 0.2444
Epoch [4/10], Step [400/938], Loss: 0.0928
Epoch [4/10], Step [500/938], Loss: 0.1394
Epoch [4/10], Step [600/938], Loss: 0.1598
Epoch [4/10], Step [700/938], Loss: 0.2146
Epoch [4/10], Step [800/938], Loss: 0.1427
Epoch [4/10], Step [900/938], Loss: 0.1729
Epoch [5/10], Step [100/938], Loss: 0.1809
Epoch [5/10], Step [200/938], Loss: 0.3656
Epoch [5/10], Step [300/938], Loss: 0.2394
Epoch [5/10], Step [400/938], Loss: 0.3710
Epoch [5/10], Step [500/938], Loss: 0.1659
Epoch [5/10], Step [600/938], Loss: 0.1906
Epoch [5/10], Step [700/938], Loss: 0.2059
Epoch [5/10], Step [800/938], Loss: 0.1552
Epoch [5/10], Step [900/938], Loss: 0.3343
Epoch [6/10], Step [100/938], Loss: 0.2181
Epoch [6/10], Step [200/938], Loss: 0.1208
Epoch [6/10], Step [300/938], Loss: 0.2252
Epoch [6/10], Step [400/938], Loss: 0.1457
Epoch [6/10], Step [500/938], Loss: 0.1276
Epoch [6/10], Step [600/938], Loss: 0.0460
Epoch [6/10], Step [700/938], Loss: 0.1754
Epoch [6/10], Step [800/938], Loss: 0.1996
Epoch [6/10], Step [900/938], Loss: 0.3706
Epoch [7/10], Step [100/938], Loss: 0.2413
Epoch [7/10], Step [200/938], Loss: 0.0577
Epoch [7/10], Step [300/938], Loss: 0.1766
Epoch [7/10], Step [400/938], Loss: 0.1948
Epoch [7/10], Step [500/938], Loss: 0.0884
Epoch [7/10], Step [600/938], Loss: 0.0870
Epoch [7/10], Step [700/938], Loss: 0.1101
Epoch [7/10], Step [800/938], Loss: 0.0858
Epoch [7/10], Step [900/938], Loss: 0.0954
Epoch [8/10], Step [100/938], Loss: 0.0930
Epoch [8/10], Step [200/938], Loss: 0.1158
Epoch [8/10], Step [300/938], Loss: 0.1354
Epoch [8/10], Step [400/938], Loss: 0.2136
Epoch [8/10], Step [500/938], Loss: 0.0833
Epoch [8/10], Step [600/938], Loss: 0.1171
Epoch [8/10], Step [700/938], Loss: 0.1069
Epoch [8/10], Step [800/938], Loss: 0.1101
Epoch [8/10], Step [900/938], Loss: 0.1135
Epoch [9/10], Step [100/938], Loss: 0.1638
Epoch [9/10], Step [200/938], Loss: 0.0749
Epoch [9/10], Step [300/938], Loss: 0.0980
Epoch [9/10], Step [400/938], Loss: 0.1275
Epoch [9/10], Step [500/938], Loss: 0.1133
Epoch [9/10], Step [600/938], Loss: 0.0825
Epoch [9/10], Step [700/938], Loss: 0.0379
Epoch [9/10], Step [800/938], Loss: 0.1294
Epoch [9/10], Step [900/938], Loss: 0.0811
Epoch [10/10], Step [100/938], Loss: 0.1760
Epoch [10/10], Step [200/938], Loss: 0.2085
Epoch [10/10], Step [300/938], Loss: 0.1033
Epoch [10/10], Step [400/938], Loss: 0.0997
Epoch [10/10], Step [500/938], Loss: 0.1419
Epoch [10/10], Step [600/938], Loss: 0.0920
Epoch [10/10], Step [700/938], Loss: 0.0780
Epoch [10/10], Step [800/938], Loss: 0.0701
Epoch [10/10], Step [900/938], Loss: 0.0602
Accuracy of the network on the 10000 test images: 96.74
```

#### Decision Tree Classifier and Support Vector Machine
```
Decision Tree: 0.8769
Support Vector Machine: 0.9792
```

#### LSTM Neural Network
```
Epoch [1/10], Step [100/938], Loss: 0.4337
Epoch [1/10], Step [200/938], Loss: 0.2637
Epoch [1/10], Step [300/938], Loss: 0.1552
Epoch [1/10], Step [400/938], Loss: 0.2181
Epoch [1/10], Step [500/938], Loss: 0.2380
Epoch [1/10], Step [600/938], Loss: 0.1099
Epoch [1/10], Step [700/938], Loss: 0.0993
Epoch [1/10], Step [800/938], Loss: 0.1525
Epoch [1/10], Step [900/938], Loss: 0.1009
Epoch [2/10], Step [100/938], Loss: 0.0430
Epoch [2/10], Step [200/938], Loss: 0.2676
Epoch [2/10], Step [300/938], Loss: 0.2343
Epoch [2/10], Step [400/938], Loss: 0.4767
Epoch [2/10], Step [500/938], Loss: 0.1476
Epoch [2/10], Step [600/938], Loss: 0.0777
Epoch [2/10], Step [700/938], Loss: 0.0345
Epoch [2/10], Step [800/938], Loss: 0.0841
Epoch [2/10], Step [900/938], Loss: 0.0867
Epoch [3/10], Step [100/938], Loss: 0.0905
Epoch [3/10], Step [200/938], Loss: 0.1358
Epoch [3/10], Step [300/938], Loss: 0.0739
Epoch [3/10], Step [400/938], Loss: 0.0190
Epoch [3/10], Step [500/938], Loss: 0.0059
Epoch [3/10], Step [600/938], Loss: 0.0225
Epoch [3/10], Step [700/938], Loss: 0.0327
Epoch [3/10], Step [800/938], Loss: 0.0527
Epoch [3/10], Step [900/938], Loss: 0.0885
Epoch [4/10], Step [100/938], Loss: 0.0852
Epoch [4/10], Step [200/938], Loss: 0.0707
Epoch [4/10], Step [300/938], Loss: 0.0789
Epoch [4/10], Step [400/938], Loss: 0.0339
Epoch [4/10], Step [500/938], Loss: 0.0284
Epoch [4/10], Step [600/938], Loss: 0.0078
Epoch [4/10], Step [700/938], Loss: 0.0210
Epoch [4/10], Step [800/938], Loss: 0.1083
Epoch [4/10], Step [900/938], Loss: 0.1679
Epoch [5/10], Step [100/938], Loss: 0.1111
Epoch [5/10], Step [200/938], Loss: 0.0776
Epoch [5/10], Step [300/938], Loss: 0.0133
Epoch [5/10], Step [400/938], Loss: 0.0176
Epoch [5/10], Step [500/938], Loss: 0.0015
Epoch [5/10], Step [600/938], Loss: 0.0269
Epoch [5/10], Step [700/938], Loss: 0.0595
Epoch [5/10], Step [800/938], Loss: 0.0194
Epoch [5/10], Step [900/938], Loss: 0.0192
Epoch [6/10], Step [100/938], Loss: 0.0087
Epoch [6/10], Step [200/938], Loss: 0.0053
Epoch [6/10], Step [300/938], Loss: 0.0170
Epoch [6/10], Step [400/938], Loss: 0.0233
Epoch [6/10], Step [500/938], Loss: 0.0662
Epoch [6/10], Step [600/938], Loss: 0.0563
Epoch [6/10], Step [700/938], Loss: 0.0077
Epoch [6/10], Step [800/938], Loss: 0.0873
Epoch [6/10], Step [900/938], Loss: 0.0802
Epoch [7/10], Step [100/938], Loss: 0.0025
Epoch [7/10], Step [200/938], Loss: 0.0499
Epoch [7/10], Step [300/938], Loss: 0.0538
Epoch [7/10], Step [400/938], Loss: 0.0035
Epoch [7/10], Step [500/938], Loss: 0.0011
Epoch [7/10], Step [600/938], Loss: 0.0099
Epoch [7/10], Step [700/938], Loss: 0.0635
Epoch [7/10], Step [800/938], Loss: 0.0060
Epoch [7/10], Step [900/938], Loss: 0.0236
Epoch [8/10], Step [100/938], Loss: 0.0014
Epoch [8/10], Step [200/938], Loss: 0.0085
Epoch [8/10], Step [300/938], Loss: 0.0296
Epoch [8/10], Step [400/938], Loss: 0.0031
Epoch [8/10], Step [500/938], Loss: 0.0087
Epoch [8/10], Step [600/938], Loss: 0.0824
Epoch [8/10], Step [700/938], Loss: 0.0162
Epoch [8/10], Step [800/938], Loss: 0.0066
Epoch [8/10], Step [900/938], Loss: 0.0043
Epoch [9/10], Step [100/938], Loss: 0.0013
Epoch [9/10], Step [200/938], Loss: 0.0250
Epoch [9/10], Step [300/938], Loss: 0.0017
Epoch [9/10], Step [400/938], Loss: 0.0228
Epoch [9/10], Step [500/938], Loss: 0.0012
Epoch [9/10], Step [600/938], Loss: 0.0106
Epoch [9/10], Step [700/938], Loss: 0.0129
Epoch [9/10], Step [800/938], Loss: 0.0255
Epoch [9/10], Step [900/938], Loss: 0.0025
Epoch [10/10], Step [100/938], Loss: 0.0025
Epoch [10/10], Step [200/938], Loss: 0.0133
Epoch [10/10], Step [300/938], Loss: 0.0135
Epoch [10/10], Step [400/938], Loss: 0.0228
Epoch [10/10], Step [500/938], Loss: 0.0004
Epoch [10/10], Step [600/938], Loss: 0.0356
Epoch [10/10], Step [700/938], Loss: 0.0061
Epoch [10/10], Step [800/938], Loss: 0.0404
Epoch [10/10], Step [900/938], Loss: 0.0056
Accuracy of the model on the test images: 98.92%
```


## Summary and Conclusion

### Problem I: Fitting a Neural Network Model

In this problem, I fit a 3-layer feed forward neural network to the data points from Homework 1. I employed two different training data sets: first 20 data points, and a combination of the first 10 and last 10 data points. The neural network attained least-square errors of 4.74 for the first training data set and 1.85 for the second training data set. When tested on the remaining data points, the network produced errors of 6.22 and 7.46, respectively.

Furthermore, I compared these results to the models fitted in Homework 1, which are linear, quadratic, and 19th-degree polynomial regression. For the first training set, the neural network outperformed the quadratic and 19th-degree polynomial models but performed worse than the linear model on the test data. For the second training set, the neural network underperformed in a similar fashion on test data. 

### Problem II: Comparison of Classifiers on the MNIST Dataset

In this problem, I analyzed the classifiers' performance (namely, LSTM, SVM, and decision tree classifiers) on the MNIST dataset by comparing the accuracy scores. I also computed the first 20 PCA modes of the digits as required.

The computed accuracy scores for the classifiers Ire as follows:

- Feed Forward Neural Network: 96.74 %
- Decision Tree: 87.69 %
- Support Vector Machine: 97.92 %
- LSTM Neural Network: 98.92 %

The LSTM neural network achieved the highest accuracy of 98.92 %, while the Decision Tree had the lowest with 87.69 %. It is also noteworthy that the feed forward network attained an accuracy score close to the SVM model.

### Conclusion

In conclusion, for the given data, the feed forward neural network performed reasonably well for the first training data set but didn't yield any discernible advantage over the linear model. For the second training data set similar results were found.

For the MNIST dataset, the LSTM neural network outperforms the other classifiers, achieving the highest accuracy in correct classification. However, the feed forward network performed almost as well as the SVM classifier and LSTM model. This indicates the LSTM's effectiveness in classifying the MNIST dataset, with the Feed Forward Neural Network as a viable alternative.

