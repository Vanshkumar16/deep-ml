import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
	"""
	Implement a position-wise feed-forward block with residual and dropout.

	Args:
		x: input vector
		W1, b1: first linear layer parameters
		W2, b2: second linear layer parameters
		dropout_p: dropout probability
		seed: random seed for reproducibility

	Returns:
		Output vector after FFN block (rounded to 4 decimals)
	"""
	# Your code heren
	np.random.seed(seed)
	x=np.array(x)
	W1=np.array(W1)
	b1=np.array(b1)
	W2=np.array(W2)
	b2=np.array(b2)

    # 1. First linear layer + ReLU
	h = np.maximum(0, W1 @ x + b1)

    # 2. Second linear layer
	y = W2 @ h + b2

    # 3. Dropout
	mask = np.random.rand(*y.shape) > dropout_p
	y = y * mask / (1 - dropout_p)

    # 4. Residual connection
	result = y + x

    # 5. Round and return as list of floats
	return [round(float(v), 4) for v in result]