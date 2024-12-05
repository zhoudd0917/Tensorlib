import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

# Check for GPU or CPU
device = torch.device("cuda" if len(sys.argv) > 1 and sys.argv[1] == "gpu" else "cpu")

# Hyperparameters
input_size = 4  # There are 4 features in the Iris dataset
hidden_size = 5
output_size = 3
learning_rate = 0.01
epochs = 1000
seed = 1234

# Load Iris dataset
iris = load_iris()
X_data = iris.data
y_data = iris.target

# Convert to one-hot encoding
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

y_train = np.eye(output_size)[y_train]
y_test = np.eye(output_size)[y_test]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train.astype(np.float32), device=device)
y_train = torch.tensor(y_train.astype(np.float32), device=device)
X_test = torch.tensor(X_test.astype(np.float32), device=device)
y_test = torch.tensor(y_test.astype(np.float32), device=device)


# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        hidden = torch.relu(self.fc1(X))
        output = self.fc2(hidden)
        return output


# Initialize the model, loss function, and optimizer
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, torch.argmax(y_train, dim=1))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.3f}", end=", ")

        # Calculate train accuracy
        _, pred_class = torch.max(y_pred, 1)
        true_class = torch.argmax(y_train, 1)
        train_accuracy.append((pred_class == true_class).float().mean().item())
        print(f"Train Accuracy: {train_accuracy[-1]:.3f}", end=", ")

        # Test
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
            loss_test = criterion(y_pred_test, torch.argmax(y_test, dim=1))
            test_loss.append(loss_test.item())

            _, pred_class = torch.max(y_pred_test, 1)
            true_class = torch.argmax(y_test, 1)
            test_accuracy.append((pred_class == true_class).float().mean().item())

        print(f"Test Loss: {loss_test.item():.3f}", end=", ")
        print(f"Test Accuracy: {test_accuracy[-1]}")

# Plotting results
if len(sys.argv) > 2:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(train_loss, label="Train Loss")
    ax[0].plot([i for i in range(0, epochs, 10)], test_loss, label="Test Loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(
        [i for i in range(0, epochs, 10)], train_accuracy, label="Train Accuracy"
    )
    ax[1].plot([i for i in range(0, epochs, 10)], test_accuracy, label="Test Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    fig.savefig(sys.argv[2])

print("Training completed.")
