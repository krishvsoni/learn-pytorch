import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    """Compute softmax for numerical stability."""
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    output = exp_x / torch.sum(exp_x, dim=1, keepdim=True)
    return output

def cross_entropy_manual(logits, targets):
    """
    Manually compute cross-entropy loss.
    
    Args:
        logits: Raw model outputs (batch_size, num_classes)
        targets: True labels (batch_size,) - class indices
    
    Returns:
        loss: Scalar cross-entropy loss
    """
    # Compute softmax probabilities
    probabilities = softmax(logits)
    
    # Get batch size
    batch_size = logits.shape[0]
    
    # Get the probability of the correct class for each sample
    # targets are indices, so we use advanced indexing
    correct_probs = probabilities[torch.arange(batch_size), targets]
    
    # Compute negative log likelihood
    neg_log_likelihood = -torch.log(correct_probs)
    
    # Average across batch
    loss = torch.mean(neg_log_likelihood)
    
    return loss

def cross_entropy_stable(logits, targets):
    """
    Numerically stable cross-entropy using log-sum-exp trick.
    
    Args:
        logits: Raw model outputs (batch_size, num_classes)
        targets: True labels (batch_size,) - class indices
    
    Returns:
        loss: Scalar cross-entropy loss
    """
    batch_size = logits.shape[0]
    
    # Log-sum-exp trick for numerical stability
    max_logits = torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    log_sum_exp = torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
    log_softmax = logits - max_logits - log_sum_exp
    
    # Get log probability of correct class
    correct_log_probs = log_softmax[torch.arange(batch_size), targets]
    
    # Compute cross-entropy loss
    loss = -torch.mean(correct_log_probs)
    
    return loss

print("=" * 60)
print("Cross-Entropy Loss Implementation")
print("=" * 60)

# Example 1: Simple 2-sample classification
print("\nExample 1: Binary Classification (2 samples, 3 classes)")
print("-" * 60)
logits = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 9.0]])
targets = torch.tensor([2, 2])  # True labels: class 2 for both

print(f"Logits (raw outputs):\n{logits}\n")
print(f"True labels: {targets}\n")

# Manual computation
loss_manual = cross_entropy_manual(logits, targets)
print(f"Manual Cross-Entropy Loss: {loss_manual.item():.4f}\n")

# Stable computation
loss_stable = cross_entropy_stable(logits, targets)
print(f"Stable Cross-Entropy Loss: {loss_stable.item():.4f}\n")

# PyTorch built-in
loss_pytorch = nn.CrossEntropyLoss()(logits, targets)
print(f"PyTorch Cross-Entropy Loss: {loss_pytorch.item():.4f}\n")

# Example 2: Multi-class classification with more samples
print("\n" + "=" * 60)
print("Example 2: Multi-class Classification (4 samples, 5 classes)")
print("-" * 60)
batch_size = 4
num_classes = 5
logits = torch.randn(batch_size, num_classes)
targets = torch.tensor([0, 2, 4, 1])  # Random true labels

print(f"Logits shape: {logits.shape}")
print(f"Logits:\n{logits}\n")
print(f"True labels: {targets}\n")

# Manual computation
loss_manual = cross_entropy_manual(logits, targets)
print(f"Manual Cross-Entropy Loss: {loss_manual.item():.4f}")

# Stable computation
loss_stable = cross_entropy_stable(logits, targets)
print(f"Stable Cross-Entropy Loss: {loss_stable.item():.4f}")

# PyTorch built-in
loss_pytorch = nn.CrossEntropyLoss()(logits, targets)
print(f"PyTorch Cross-Entropy Loss: {loss_pytorch.item():.4f}\n")

# Example 3: Comparison with different probability distributions
print("\n" + "=" * 60)
print("Example 3: Softmax Probabilities & Loss Interpretation")
print("-" * 60)
logits = torch.tensor([[2.0, 1.0, 0.1]])
targets = torch.tensor([0])

probs = softmax(logits)
loss = cross_entropy_manual(logits, targets)

print(f"Logits: {logits[0]}")
print(f"Softmax Probabilities: {probs[0]}")
print(f"True class: 0")
print(f"Probability of correct class: {probs[0, 0].item():.4f}")
print(f"Cross-Entropy Loss: {loss.item():.4f}")
print(f"(-log({probs[0, 0].item():.4f})) = {-torch.log(probs[0, 0]).item():.4f}")
