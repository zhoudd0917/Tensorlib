import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Device selection
device = torch.device("cuda" if len(sys.argv) > 1 and sys.argv[1] == "gpu" else "cpu")

input_size = 28 * 28
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 200
seed = 1234

# Load MNIST dataset
mnist = fetch_openml("mnist_784", parser="auto")
X_data = mnist.data.to_numpy()
y_data = mnist.target.to_numpy().astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# Standardize pixel values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(-1, input_size).astype(np.float32)
X_test = X_test.reshape(-1, input_size).astype(np.float32)

# One-hot encode labels
y_train = np.eye(output_size)[y_train]
y_test = np.eye(output_size)[y_test]

# Convert to torch tensors
X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
y_train = torch.tensor(y_train, device=device, dtype=torch.float32)
X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
y_test = torch.tensor(y_test, device=device, dtype=torch.float32)

# Create DataLoader for batching
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_data,
    batch_size=len(train_data),
    shuffle=True,
    num_workers=4,
)
test_loader = DataLoader(
    test_data,
    batch_size=len(test_data),
    shuffle=False,
    num_workers=4,
)


# Initialize the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = torch.relu(self.fc1(x))
        output = self.fc2(hidden)
        return output


model = SimpleNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Tracking metrics
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, axis=1))

        # Backward pass
        loss.backward()

        # Gradient descent
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, axis=1)).sum().item()

    train_loss.append(running_loss / len(train_loader))
    train_accuracy.append(correct / total)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {train_loss[-1]:.3f}, ", end="")
        print(f"Train Accuracy: {train_accuracy[-1]:.3f}, ", end="")

        # Test
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, torch.argmax(labels, axis=1))
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, axis=1)).sum().item()

        test_loss.append(running_loss / len(test_loader))
        test_accuracy.append(correct / total)

        print(f"Test Loss: {test_loss[-1]:.3f}, ", end="")
        print(f"Test Accuracy: {test_accuracy[-1]:.3f}")

# Plot results if file output is provided
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
