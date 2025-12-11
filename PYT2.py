import torch
import torch.nn as nn
import torch.optim as optim

# 1. Create dummy input and output data
X = torch.randn(100, 1)          # 100 samples, 1 feature
y = 3 * X + 2 + 0.1 * torch.randn(100, 1)   # Linear relation with noise

# 2. Build a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)   # 1 input → 1 output

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# 3. Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training loop
for epoch in range(300):
    y_pred = model(X)             # Forward pass
    loss = criterion(y_pred, y)   # Compute loss

    optimizer.zero_grad()         # Clear old gradients
    loss.backward()               # Backpropagation
    optimizer.step()              # Update weights

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")

# 5. Test the trained model
test_value = torch.tensor([[5.0]])
prediction = model(test_value)
print("Prediction for input 5:", prediction.item())
