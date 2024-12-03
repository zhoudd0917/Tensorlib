import tensorlib as tl
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = tl.Device.CPU

# Hyperparameters
input_size = 4  # There are 4 features in the Iris dataset
hidden_size = 5
output_size = 3
learning_rate = 0.01
epochs = 100

# Load Iris dataset
iris = load_iris()
X_data = iris.data
y_data = iris.target

# Convert to binary classification: class 0 (Setosa) vs class 1 (Versicolor)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

y_train = np.eye(output_size)[y_train]
y_test = np.eye(output_size)[y_test]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X = tl.Tensor(X_train.astype(np.float32), device=device, requires_grad=False)
y = tl.Tensor(
    y_train.astype(np.float32).reshape(-1, output_size),
    device=device,
    requires_grad=False,
)

X_test = tl.Tensor(X_test.astype(np.float32), device=device, requires_grad=False)
y_test = tl.Tensor(
    y_test.astype(np.float32).reshape(-1, output_size),
    device=device,
    requires_grad=False,
)

W1 = tl.randn([input_size, hidden_size], requires_grad=True)
b1 = tl.zeros([hidden_size], requires_grad=True)
W2 = tl.randn([hidden_size, output_size], requires_grad=True)
b2 = tl.zeros([output_size], requires_grad=True)


def forward(X):
    hidden = tl.relu(X @ W1 + b1)
    output = hidden @ W2 + b2
    return output


train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(epochs):
    y_pred = forward(X)
    loss = tl.mean(tl.cross_entropy(y_pred, y))

    loss.backward()

    # Gradient descent
    W1 -= learning_rate * W1.grad
    b1 -= learning_rate * b1.grad
    W2 -= learning_rate * W2.grad
    b2 -= learning_rate * b2.grad

    # Clear gradients
    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()

    train_loss.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.3f}", end=", ")
        pred_class = tl.argmax(y_pred, axis=1).to_numpy().astype(int)
        true_class = tl.argmax(y, axis=1).to_numpy().astype(int)
        train_accuracy.append(np.mean(pred_class == true_class))
        print(f"Train Accuracy: {train_accuracy[-1]:.3f}", end=", ")

        # Test
        y_pred = forward(X_test)
        loss = tl.mean(tl.cross_entropy(y_pred, y_test))
        test_loss.append(loss.item())
        print(f"Test Loss: {loss.item():.3f}", end=", ")
        pred_class = tl.argmax(y_pred, axis=1).to_numpy().astype(int)
        true_class = tl.argmax(y_test, axis=1).to_numpy().astype(int)
        test_accuracy.append(np.mean(pred_class == true_class))
        print(f"Test Accuracy: {test_accuracy[-1]}")

if len(sys.argv) > 1:
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
    fig.savefig(sys.argv[1])

print("Training completed.")
