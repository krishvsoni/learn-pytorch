# softmax usecase here is to convert the output of a neural network into probabilities for classification tasks.
# The softmax function takes a vector of raw scores (logits) and transforms them into probabilities that sum to 1. This is particularly useful in multi-class classification problems where we want to assign probabilities to each class.
#logits = model(X)  # Output from the neural network (before softmax)
#probabilities = doftmax(logits)  # Convert logits to probabilities using softmax
# logits in simple terms are the raw outputs from the last layer of a neural network, which can be any real numbers. 
# The softmax function transforms these logits into probabilities that sum to 1, making it easier to interpret the model's predictions as confidence levels for each class.

import torch
import torch.nn as nn
import numpy as np

def doftmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    output = exp_x / torch.sum(exp_x, dim=1, keepdim=True)
    return output
print("=" * 60)
print("Testing Softmax Function")
print("=" * 60)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 9.0]])
softmax_output = doftmax(x)
print(f"Input:\n{x}\n")
print(f"Softmax Output:\n{softmax_output}\n")
print(f"Sum across classes (should be 1):\n{torch.sum(softmax_output, dim=1)}")
