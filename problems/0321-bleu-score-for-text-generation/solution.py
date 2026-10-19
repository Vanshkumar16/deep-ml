import torch
import math
from collections import Counter

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences
    using PyTorch tensor operations for numerical computations.
    
    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        BLEU score between 0 and 1 as a Python float
    """
    # Your code here
    if len(candidate) == 0:
        return 0.0

    candidate_len = len(candidate)

    # Find closest reference length
    reference_len = min(
        [len(ref) for ref in references],
        key=lambda x: (abs(x - candidate_len), x)
    )

    precision_scores = []

    for n in range(1, max_n + 1):

        # Candidate n-grams
        candidate_ngrams = Counter(
            tuple(candidate[i:i+n])
            for i in range(len(candidate) - n + 1)
        )

        # Maximum count of each n-gram among references
        max_reference_counts = {}

        for reference in references:

            reference_ngrams = Counter(
                tuple(reference[i:i+n])
                for i in range(len(reference) - n + 1)
            )

            for gram, count in reference_ngrams.items():
                max_reference_counts[gram] = max(
                    max_reference_counts.get(gram, 0),
                    count
                )

        # Clipped count
        clipped_count = 0

        for gram, count in candidate_ngrams.items():
            clipped_count += min(
                count,
                max_reference_counts.get(gram, 0)
            )

        total_count = sum(candidate_ngrams.values())

        if total_count == 0:
            return 0.0

        precision_n = clipped_count / total_count

        if precision_n == 0:
            return 0.0

        precision_scores.append(precision_n)

    # Convert precisions to PyTorch tensor
    precision_tensor = torch.tensor(
        precision_scores,
        dtype=torch.float32
    )

    # Geometric mean
    geometric_mean = torch.exp(
        torch.mean(torch.log(precision_tensor))
    ).item()

    # Brevity penalty
    if candidate_len > reference_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(
            1 - reference_len / candidate_len
        )

    return brevity_penalty * geometric_mean