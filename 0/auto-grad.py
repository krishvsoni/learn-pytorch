import torch

x = torch.randn(3, requires_grad=True)
print(x)

y=x+2
print(y)


z=y*y*2
z=z.mean()

print(z)

#gradient is dz/dx = dz/dy * dy/dx 
# tldr  how much z changes with respect to x
#backward() computes the gradient of z with respect to x


v=torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
z.backward() # i.e dz/dx = 0.1*1.5
print(x.grad)



# z.backward() # i.e dz/dx = 1.5
# print(x.grad)