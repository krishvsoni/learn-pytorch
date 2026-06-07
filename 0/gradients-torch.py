# use torch for numeric ops
import torch
# f = w*x
# w = 2.0
# x = 3.0
# f = 6.0

#y=2x

x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)

# weight (float)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction 

def forward(x):
    return w*x

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

print("Predicted value:", forward(x), "Loss:", loss(y, forward(x)).item())

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # forward pass
    y_predicted = forward(x)
    # compute loss
    l = loss(y, y_predicted)
    l.backward()  # compute gradients using autograd
    
    # update weights using computed gradient
    with torch.no_grad():  # disable gradient tracking for weight update
        w -= learning_rate * w.grad
        w.grad.zero_()  # zero out gradients for next iteration
    
    print(f"Epoch {epoch}: w = {w:.3f}, loss = {l:.3f}")

print(f"Final prediction after training: f(5) = {forward(torch.tensor(5.)).item():.3f}")