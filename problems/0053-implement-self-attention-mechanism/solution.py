import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    d_k=Q.shape[1]
    attention_score=np.matmul(Q, K.T)/np.sqrt(d_k)
    exp_x = np.exp(attention_score - np.max(attention_score,axis=-1, keepdims=True))
    attention_weight= exp_x / np.sum(exp_x,axis=-1, keepdims=True)
    # attention_weight=np.softmax(attention_score, dim=1)
    output=np.matmul(attention_weight, V)
    return output
    
