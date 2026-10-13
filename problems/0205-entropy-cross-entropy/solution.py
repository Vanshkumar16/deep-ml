import torch

def entropy_and_cross_entropy(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution (torch.Tensor)
        Q: Predicted probability distribution (torch.Tensor)
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    # Your code here
    eps = 1e-9
    P = torch.as_tensor(P, dtype=torch.float32)
    Q = torch.as_tensor(Q, dtype=torch.float32)
    P = torch.clamp(P, min=eps, max=1.0)
    Q = torch.clamp(Q, min=eps, max=1.0)
    entropy=-torch.sum(P*torch.log(P))
    cross_entropy=-torch.sum(P*torch.log(Q))
    return entropy.item(),cross_entropy.item()