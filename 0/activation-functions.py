import torch
import torch.nn as nn
import torch.nn.functional as F


# Activation Functions
def apply_relu(x):
    relu = nn.ReLU()
    return relu(x)


def apply_sigmoid(x):
    sigmoid = nn.Sigmoid()
    return sigmoid(x)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        # Linear Layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        # Activation Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply linear1 + relu activation
        x = self.relu(self.linear1(x))  # Activation function (ReLU)
        # Apply linear2 + sigmoid activation
        x = self.sigmoid(self.linear2(x))  # Output layer (Sigmoid)
        return x


# Example usage:
if __name__ == "__main__":
    # Initialize the network
    # Parameters: input_size, hidden_size, output_size
    model = NeuralNetwork(input_size=10, hidden_size=5, output_size=1)
    
    # Create sample input (batch_size=2, input_size=10)
    input_data = torch.randn(2, 10)
    
    # Forward pass
    output = model(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Test individual activation functions
    test_tensor = torch.tensor([[-1.0, 2.0, -0.5], [0.1, -3.0, 1.5]])
    print(f"\nOriginal tensor:\n{test_tensor}")
    print(f"After ReLU:\n{apply_relu(test_tensor)}")
    print(f"After Sigmoid:\n{apply_sigmoid(test_tensor)}")