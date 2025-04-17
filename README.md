# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Yogesh rao S D

### Register Number: 212222110055
```python


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.arange(1, 51, dtype=np.float32).reshape(-1, 1)
y = 3.5 * x + np.random.normal(0, 5, size=x.shape)

x_train = torch.tensor(x, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

model = Model(in_features=1, out_features=1)


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 100
loss_values = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_values, label="Training Loss", color='b')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.legend()
plt.show()



model.eval()
with torch.no_grad():
    y_pred = model(x_train).numpy()
plt.figure(figsize=(8, 5))
plt.scatter(x, y, label="Data", color='')
plt.plot(x, y_pred, label="Best Fit Line", color='black')
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.title("Best Fit Line for Regression")
plt.legend()
plt.show()

new_sample = torch.tensor([[55.0]])
predicted_output = model(new_sample).item()
print(f"Prediction for input 55: {predicted_output:.2f}")
```

### Dataset Information
![image](https://github.com/user-attachments/assets/4b174392-c01e-40b0-ab9e-ef32371bbafe)

### OUTPUT
Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/ecb45570-f01b-4c8b-a5d6-2a0f32058b47)

Best Fit line plot

![image](https://github.com/user-attachments/assets/a36623f7-3511-494f-9358-baa6151ddcba)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/df3bf07c-4bf5-4ae4-89af-4f80bfc6b27d)


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
