import torch
import torch.nn as nn

"""
3 DIFFERENT METHODS TO REMEMBER:
- torch.save(arg, PATH)
- torch.load(PATH)
- model.load_state_dict(arg)
"""

"""
2 DIFFERENT WAYS OF SAVING

1) Save whole model (not recommended)

torch.save(model, PATH)

model = torch.load(PATH, weights_only=False)
model.eval()

2) Save only state_dict (recommended)

torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""


class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Create model
model = Model(n_input_features=6)



print("Original Model Parameters:")
for param in model.parameters():
    print(param)

FILE = "model_complete.pth"

# Save complete model
torch.save(model, FILE)

# Load complete model
# PyTorch 2.6+ requires weights_only=False
loaded_model = torch.load(FILE, weights_only=False)
loaded_model.eval()

print("\nLoaded Model Parameters:")
for param in loaded_model.parameters():
    print(param)



FILE = "model_state_dict.pth"

# Save only weights
torch.save(model.state_dict(), FILE)

print("\nSaved State Dict:")
print(model.state_dict())

# Recreate model architecture
loaded_model = Model(n_input_features=6)

# Load weights
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

print("\nLoaded State Dict:")
print(loaded_model.state_dict())


learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
}

print("\nOriginal Optimizer State:")
print(optimizer.state_dict())

FILE = "checkpoint.pth"

# Save checkpoint
torch.save(checkpoint, FILE)

# Create fresh model and optimizer
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

# Load checkpoint
checkpoint = torch.load(FILE)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])
epoch = checkpoint["epoch"]

# For inference
model.eval()

# If continuing training:
# model.train()

print("\nLoaded Optimizer State:")
print(optimizer.state_dict())

print(f"\nResumed from epoch: {epoch}")



"""
# 1) Save on GPU, Load on CPU

device = torch.device("cuda")
model.to(device)

torch.save(model.state_dict(), "gpu_model.pth")

device = torch.device("cpu")

model = Model(n_input_features=6)
model.load_state_dict(
    torch.load("gpu_model.pth", map_location=device)
)

# ------------------------------------------------------------

# 2) Save on GPU, Load on GPU

device = torch.device("cuda")
model.to(device)

torch.save(model.state_dict(), "gpu_model.pth")

model = Model(n_input_features=6)
model.load_state_dict(torch.load("gpu_model.pth"))
model.to(device)


# 3) Save on CPU, Load on GPU

torch.save(model.state_dict(), "cpu_model.pth")

device = torch.device("cuda")

model = Model(n_input_features=6)
model.load_state_dict(
    torch.load("cpu_model.pth", map_location="cuda:0")
)
model.to(device)
"""

print("\nDone!")