import torch

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    probabilities=torch.tensor(probabilities,dtype=torch.float32)
    log_proba=-torch.mean(torch.log(probabilities))
    perplexity=torch.exp(log_proba)
    return perplexity.item()