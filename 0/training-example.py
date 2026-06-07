import torch 
weights = torch.ones(4, requires_grad=True)


for epoch in range(81):
    model_output=(weights*3).sum() 

    model_output.backward() # computes the gradient of model_output with respect to weights
    print(weights.grad) # prints the gradient of weights

    weights.grad.zero_() # resets the gradient to zero for the next iteration