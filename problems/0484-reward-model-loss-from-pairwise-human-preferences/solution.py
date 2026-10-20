import numpy as np
import math

def reward_model_loss(chosen_rewards: list, rejected_rewards: list, margin: float = 0.0) -> dict:
    """
    Compute reward model loss from pairwise human preferences.
    
    Args:
        chosen_rewards: Reward scores for preferred responses
        rejected_rewards: Reward scores for non-preferred responses
        margin: Minimum desired gap between chosen and rejected scores
    
    Returns:
        Dictionary with 'loss' (float) and 'accuracy' (float)
    """
    # score_diff=chosen_rewards-rejected_rewards
    # log_val=np.log(score_diff)
    # avg_loss=-np.mean(log_val)
    # return {
    #     'loss':avg_loss,'accuracy':
    # }
    losses = []
    correct = 0

    for c, r in zip(chosen_rewards, rejected_rewards):
        # Score difference with margin
        diff = c - r - margin

        # Numerically stable: -log(sigmoid(diff))
        loss = max(0.0, -diff) + math.log1p(math.exp(-abs(diff)))
        losses.append(loss)

        # Accuracy uses the actual rewards, not the margin-adjusted difference
        if c > r:
            correct += 1

    avg_loss = sum(losses) / len(losses)
    accuracy = correct / len(chosen_rewards)

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4)
    }
