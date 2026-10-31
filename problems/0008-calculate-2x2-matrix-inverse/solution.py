def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]
    
    # Calculate the determinant (ad - bc)
    det = a * d - b * c
    
    # If the determinant is zero, the matrix is not invertible
    if det == 0:
        return None
        
    # Apply the inverse formula for a 2x2 matrix
    inv_det = 1 / det
    return [
        [d * inv_det, -b * inv_det],
        [-c * inv_det, a * inv_det]
    ]