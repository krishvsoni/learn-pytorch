import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


INPUT_DIM = 20
HIDDEN_DIM = 64
NUM_CLASSES = 2

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.randn(1000, INPUT_DIM)
y = torch.randint(0, NUM_CLASSES, (1000,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet().to(DEVICE)


criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(
    model.parameters(),
    lr=LR
)


scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for x_batch, y_batch in loader:

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(x_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {total_loss/len(loader):.4f} "
        f"LR: {current_lr:.8f}"
    )

print("Training Complete")