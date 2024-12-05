import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Device selection
device = torch.device("cuda" if len(sys.argv) > 1 and sys.argv[1] == "gpu" else "cpu")

input_size = 28 * 28
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 200
seed = 1234

# Set random seed
torch.manual_seed(seed)

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

# Initialize weights and biases
W1 = torch.randn(input_size, hidden_size, device=device, requires_grad=True)
b1 = torch.zeros(hidden_size, device=device, requires_grad=True)
W2 = torch.randn(hidden_size, output_size, device=device, requires_grad=True)
b2 = torch.zeros(output_size, device=device, requires_grad=True)


# Forward pass
def forward(X):
    hidden = torch.relu(X @ W1 + b1)
    output = hidden @ W2 + b2
    return output


# Tracking metrics
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = forward(X_train)
    loss = torch.mean(torch.sum(-y_train * torch.log_softmax(y_pred, dim=1), dim=1))
    train_loss.append(loss.item())

    # Backward pass
    loss.backward()

    # Gradient descent
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

        # Zero gradients
        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()

    # Calculate training accuracy
    _, predicted = torch.max(y_pred, 1)
    _, true_labels = torch.max(y_train, 1)
    train_accuracy.append((predicted == true_labels).sum().item() / len(y_train))

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {train_loss[-1]:.3f}, ", end="")
        print(f"Train Accuracy: {train_accuracy[-1]:.3f}, ", end="")

        # Test
        with torch.no_grad():
            y_pred = forward(X_test)
            loss = torch.mean(
                torch.sum(-y_test * torch.log_softmax(y_pred, dim=1), dim=1)
            )
            test_loss.append(loss.item())

            _, predicted = torch.max(y_pred, 1)
            _, true_labels = torch.max(y_test, 1)
            test_accuracy.append((predicted == true_labels).sum().item() / len(y_test))

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
