import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    w=initial_weights.clone().detach().float().requires_grad_(True)
    # w.requires_grad_(True)
    b=torch.tensor(initial_bias,dtype=torch.float32,requires_grad=True)
    mse_values=[]
    for epoch in range(epochs):
        #forward pass
        z=features @ w + b
        #sigmoid function
        pred=torch.sigmoid(z)
        #mse loss 
        loss=torch.mean((pred-labels)**2)
        #record
        mse_values.append(round(loss.item(),4))
        #backpropagation
        loss.backward()
        # gradient descent
        with torch.no_grad():
            w-=learning_rate * w.grad
            b-=learning_rate * b.grad

        w.grad.zero_()
        b.grad.zero_()
    
    return (
            [round(i.item(), 4) for i in w],
            round(b.item(), 4),
            mse_values
        )
