import torch
import numpy as np
x = torch.rand(3, 4)
y = torch.rand(3, 4)

# print(x[0])

# # print(x)
# # print(y)

# # z = x + y
# # print(z)

# k = torch.ones(3)
# print("k:", k)
# print(k)
# u=k.numpy()
# print("u:", u)
# print(u)

# k.add_(1)
# print(k)
# print(u)

# print("p:", p)
# p = torch.from_numpy(u)
# print("p:", p)  

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    np.random.seed(0)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))

wel = torch.ones(5,requires_grad=True)
print(wel)