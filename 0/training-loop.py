#design model (input,output size, forward pass)
#construct loss and optimizer
#training loop
# - forward pass: compute prediction and loss
# - backward pass: compute gradients
# - update weights




import torch
import torch.nn as nn
import torch.optim as optim
x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_samples, n_features = x.shape
print(n_samples, n_features)

model=nn.Linear(in_features=1, out_features=1) # model = w*x + b


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_dim=n_features, output_dim=1)
# Loss function
loss = nn.MSELoss()

w_init = model.linear.weight.item()
b_init = model.linear.bias.item()
print(f"Initial - w = {w_init:.3f}, b = {b_init:.3f}")
print("Predicted value:", model(x).squeeze(), "Loss:", loss(y, model(x)).item())

# Training
learning_rate = 0.01
n_iters = 100

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # forward pass
    y_predicted = model(x)
    # compute loss
    l = loss(y, y_predicted)
    l.backward()  # compute gradients using autograd
    optimizer.step()  # update weights using computed gradient
    optimizer.zero_grad()  # zero out gradients for next iteration

    w_val = model.linear.weight.item()
    b_val = model.linear.bias.item()
    print(f"Epoch {epoch}: w = {w_val:.3f}, b = {b_val:.3f}, loss = {l:.3f}")

w_final = model.linear.weight.item()
b_final = model.linear.bias.item()
print(f"Final - w = {w_final:.3f}, b = {b_final:.3f}")
print(f"Final prediction after training: f(5) = {model(torch.tensor([[5.]])).item():.3f}")