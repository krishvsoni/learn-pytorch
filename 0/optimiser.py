import torch

weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD([weights], lr=0.01)

# Training data
X = torch.randn(10, 4)
y = torch.randn(10, 1)

# Forward pass
output = X @ weights.unsqueeze(1)
loss = torch.nn.functional.mse_loss(output, y)

# Backward pass
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"Weights before step: {weights.data}")

optimizer.step()  # performs a single optimization step (parameter update)

print(f"Weights after step: {weights.data}")

optimizer.zero_grad()  # clears the gradients of all optimized parameters