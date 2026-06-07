import numpy as np

# f = w*x
# w = 2.0
# x = 3.0
# f = 6.0

#y=2x
x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

w=0.0

# model prediction 

def forward(x):
    return w*x

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

#gradient
# MSE = 1/N * sum((y_predicted - y)**2)
# dJ/dw = 1/N * sum(2x(y_predicted - y

def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print("Predicted value:", forward(x), "Loss:", loss(y, forward(x)))

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # forward pass
    y_predicted = forward(x)
    # compute loss
    l = loss(y, y_predicted)
    # compute gradients
    dw = gradient(x, y, y_predicted)
    # update weights
    w -= learning_rate * dw
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: w = {w:.3f}, loss = {l:.3f}")
        print(f"Final prediction after training: f(5) = {forward(5):.3f}")