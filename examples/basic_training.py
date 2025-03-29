import numpy as np
from pyraml.core import Tensor
from pyraml.nn.layers.linear import Linear
from pyraml.nn.losses import MSELoss
from pyraml.optim.sgd import SGD

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 2)  # 100 samples, 2 features
y = 2 * X[:, 0] - 3 * X[:, 1] + 1 + np.random.randn(100) * 0.1  # y = 2x1 - 3x2 + 1 + noise

# Convert to PyraML tensors
X_tensor = Tensor(X)
y_tensor = Tensor(y.reshape(-1, 1))

# Create model
model = Linear(in_features=2, out_features=1)

# Define loss function and optimizer
criterion = MSELoss()
optimizer = SGD(parameters=[model.weight, model.bias], lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data.item():.4f}")

# Test the model
test_X = np.array([[1.0, 2.0]])
test_tensor = Tensor(test_X)
prediction = model(test_tensor)
print(f"\nTest prediction for input {test_X}:")
print(f"Predicted value: {prediction.data.item():.4f}")
