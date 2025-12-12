import torch
import torch.nn as nn

#Sample input and target output data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]]) # Feature input
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]]) # Target output

# Define a simple linear regression model: y = wx + b
model = nn.Linear(in_features=1, out_features=1)

# Define the Mean Squared Error loss function
criterion = nn.MSELoss()

# Define the optimizer: Stochastic gradient decent optimizer to update weights
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Training loop
for epoch in range(10000): # Run for 1000 epochs
    y_pred = model(X) # Forward pass: predict output


    loss = criterion(y_pred, Y) # Compute MSE Loss

    optimizer.zero_grad() # Clear previous gradients

    loss.backward() # Backpropagation: compute gradients

    optimizer.step() # Update waights using gradients

    # Print progree every 100 epochs

    if (epoch+1) % 1000 == 0:
        print(f'Epoch{epoch+1}, Loss: {loss.item():.4f}')

    # After training, print the learned waight and bias
    params = list(model.parameters())
    print(f'Learned weight: {params[0].item():.4f}, bias: {params[1].item():.4f}')
    # Test the model with a new input