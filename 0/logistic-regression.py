#design model (input,output size, forward pass)
#construct loss and optimizer
#training loop
# - forward pass: compute prediction and loss
# - backward pass: compute gradients
# - update weights


import torch
import torch.nn as nn
import torch.optim as optim

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# prepare data
# binary classification dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
print(f"Dataset samples: {n_samples}, features: {n_features}")

# standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# convert to torch tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float().unsqueeze(1)
y_test = torch.from_numpy(y_test).float().unsqueeze(1)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = LogisticRegressionModel(n_features)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

n_epochs = 1000
losses = []

for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 100 == 0 or epoch == 1:
        with torch.no_grad():
            preds = torch.sigmoid(logits)
            predicted_labels = (preds >= 0.5).float()
            accuracy = (predicted_labels == y_train).float().mean().item()
        print(f"Epoch {epoch:4d}/{n_epochs}, loss={loss.item():.4f}, train_acc={accuracy:.4f}")

model.eval()
with torch.no_grad():
    train_logits = model(X_train)
    test_logits = model(X_test)
    train_preds = (torch.sigmoid(train_logits) >= 0.5).float()
    test_preds = (torch.sigmoid(test_logits) >= 0.5).float()
    train_accuracy = (train_preds == y_train).float().mean().item()
    test_accuracy = (test_preds == y_test).float().mean().item()

print(f"Final train accuracy: {train_accuracy:.4f}")
print(f"Final test accuracy:  {test_accuracy:.4f}")

# plot the training loss curve
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Logistic Regression Training Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show() 