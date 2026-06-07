import torch

x = torch.randn(3, requires_grad=True)
print("before gradient history:",x)
# option 1 requires_grad=False
# option 2 x.detach()  # returns a new tensor that shares the same data but does not require gradients
# option 3 with torch.no_grad():  # context manager that temporarily sets all the requires

x.requires_grad_(False)  # in-place operation that sets requires_grad to False
print(x)

y=x.detach()
print(y)

with torch.no_grad():
    y=x+2
    print(y)

# the use of detach() and no_grad() is to prevent the tracking of operations for gradient computation, which can save memory and computational resources when gradients are not needed.