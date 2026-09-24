import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    logits=torch.tensor(scores,dtype=torch.float32)
    # prob=softmax(logits)
    maxi=torch.max(logits)
    expo=torch.exp(logits-maxi)
    sum_expo=torch.sum(expo)
    return (expo/sum_expo).tolist()
