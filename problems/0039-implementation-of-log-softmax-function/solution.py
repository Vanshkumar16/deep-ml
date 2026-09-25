import torch
import torch.nn
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    # Your code here
    z=torch.tensor(scores,dtype=torch.float32)
    # result=torch.l(z)
    return torch.nn.functional.log_softmax(z)
