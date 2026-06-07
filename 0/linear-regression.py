#design model (input,output size, forward pass)
#construct loss and optimizer
#training loop
# - forward pass: compute prediction and loss
# - backward pass: compute gradients
# - update weights


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from  sklearn import datasets
import matplotlib.pyplot as plt

# prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1)
n_samples, n_features = X.shape
print(n_samples, n_features)

# model -> loss and optimizer -> training loop

model=nn.Linear(in_features=n_features, out_features=1) # model = w*x + b 
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)


criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # backward pass
    loss.backward()
    
    # update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


#plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro', label='Original data')
plt.plot(X_numpy, predicted, label='Fitted line')
plt.legend()
plt.show()
