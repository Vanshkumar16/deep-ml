import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	arr=np.array(matrix)
	if mode == 'column':
		return  arr.mean(axis=0).tolist()
	return arr.mean(axis=1).tolist()