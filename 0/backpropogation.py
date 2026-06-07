# chain rule 
# dx/dt = dx/dy * dy/dt
# dz/dt = dz/dy * dy/dt
# dz/dt = dz/dy * dy/dt = dz/dy * dx/dt * dt/dt = dz/dy * dx/dt
# dz/dt = dz/dy * dx/dt
# dz/dt = dz/dy * dx/dt = dz/dy * dx/dt * dt/dt = dz/dy * dx/dt

# computational graph
# x*y=c. ; f=xy

#step 1 is forward pass: compute the loss
#step 2 compute local gradients: df/dx = y, df/dy = x
# step 3 is backward pass: compute loss dloss/dweights using the chain rule: dloss/dweights = dloss/df * df/dweights
# step 4 update weights: weights = weights - learning_rate * dloss/dweights


import torch

x=torch.tensor(1.0, )
y=torch.tensor(3.0, )

w=torch.tensor(2.0, requires_grad=True)


# forward pass compute the loss
y_hat=w*x
loss_= (y_hat-y)**2 # squared error loss
print(loss_)

# backward pass compute the gradients
loss_.backward()
w.grad
print(w.grad)

# update weights ; next step in the training loop
learning_rate=0.01
w=w-learning_rate*w.grad
print(w)