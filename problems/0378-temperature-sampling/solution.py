import torch
import torch.nn.functional as F

def temperature_sampling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Compute temperature-scaled softmax probabilities from logits.
    
    Args:
        logits: 1D torch.Tensor of raw model output scores
        temperature: float controlling distribution sharpness
    
    Returns:
        torch.Tensor of probabilities after temperature scaling
    """
    # Your code here
    logits = torch.from_numpy(logits).float() if not isinstance(logits, torch.Tensor) else logits.float()
    if temperature <= 0:
        probabilities = torch.zeros_like(logits, dtype=torch.float32)
        probabilities[torch.argmax(logits)] = 1.0
        return probabilities
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=0)

    return probabilities
    # if temperature<=0:
    #     probabilities = torch.zeros_like(logits, dtype=torch.float32)
    #     probabilities[torch.argmax(logits)] = 1.0
    #     return probabilities.tolist()
    # ans=logits/temperature
    # maxi=torch.softmax(ans, dim=0)
    # return maxi.tolist()
    # logits=torch.from_numpy(logits)
    # if temperature <= 0:
    #     probabilities = torch.zeros_like(logits, dtype=torch.float32)
    #     probabilities[torch.argmax(logits)] = 1.0
    #     return probabilities.tolist()

    # scaled_logits = logits / temperature
    # scaled_logits=scaled_logits-torch.max(scaled_logits)
    # probabilities=torch.exp(scaled_logits)

    # return probabilities.tolist()