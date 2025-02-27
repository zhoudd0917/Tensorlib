import tensorlib as tl
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# Device selection
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    device = tl.Device.GPU
else:
    device = tl.Device.CPU

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

# Convert to tensorlib tensors
X_train = tl.Tensor(X_train, device=device, requires_grad=False)
y_train = tl.Tensor(y_train, device=device, requires_grad=False)
X_test = tl.Tensor(X_test, device=device, requires_grad=False)
y_test = tl.Tensor(y_test, device=device, requires_grad=False)

# Initialize weights and biases
W1 = tl.randn(
    [input_size, hidden_size], seed=seed, device=device, requires_grad=True
) * np.sqrt(2 / input_size)
b1 = tl.zeros([hidden_size], device=device, requires_grad=True)
W2 = tl.randn(
    [hidden_size, output_size], seed=seed, device=device, requires_grad=True
) * np.sqrt(2 / hidden_size)
b2 = tl.zeros([output_size], device=device, requires_grad=True)


# Forward pass
def forward(X):
    hidden = tl.relu(X @ W1 + b1)
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
    loss = tl.mean(tl.cross_entropy(y_pred, y_train))

    # Backward pass
    loss.backward()

    # Gradient descent
    with tl.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

    train_loss.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.3f}, ", end="")
        pred_class = tl.argmax(y_pred, axis=1).to_numpy().astype(int)
        true_class = tl.argmax(y_train, axis=1).to_numpy().astype(int)
        train_accuracy.append(np.mean(pred_class == true_class))
        # train accuracy to 3 decimal places
        print(f"Train Accuracy: {train_accuracy[-1]:.3f}, ", end="")

        # Test
        y_pred = forward(X_test)
        loss = tl.mean(tl.cross_entropy(y_pred, y_test))
        test_loss.append(loss.item())
        print(f"Test Loss: {loss.item():.3f}, ", end="")
        pred_class = tl.argmax(y_pred, axis=1).to_numpy().astype(int)
        true_class = tl.argmax(y_test, axis=1).to_numpy().astype(int)
        test_accuracy.append(np.mean(pred_class == true_class))
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
