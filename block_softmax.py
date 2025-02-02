import numpy as np
import pandas as pd
from datetime import datetime
from scipy.special import softmax

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={"float": lambda x: f"{x:04.2f}"})
pd.options.display.float_format = "{:04.5f}".format


def block_softmax(S: np.ndarray, block_size: int) -> np.ndarray:
    """
    Computes row-wise softmax in blocks to mimic FlashAttention's strategy.

    Parameters:
    - S: (L, L) Attention score matrix
    - block_size: Size of each block (assumed to divide L evenly for simplicity)

    Returns:
    - P: (L, L) Softmax probability matrix
    """
    L = len(S)
    P = np.zeros_like(S)  # Output probability matrix
    row_max = np.full((L,), -np.inf)  # Store row-wise max for stability
    row_sum_exp = np.zeros((L,))  # Accumulate denominator Z incrementally

    # Process each block sequentially
    for start in range(0, L, block_size):
        end = min(start + block_size, L)
        S_block = S[:, start:end]  # Extract block

        # Step 1: Compute local max per row and update global max
        local_max = np.max(S_block, axis=1, keepdims=True)
        row_max = np.maximum(row_max, local_max.flatten())  # Update global max

        # Step 2: Compute exponentials in a numerically stable way
        S_block_exp = np.exp(S_block - row_max[:, None])  # Subtract global max

        # Step 3: Update row-wise softmax denominator (Z)
        row_sum_exp += np.sum(S_block_exp, axis=1)  # Accumulate denominator

        # Store temporarily (final normalization is deferred)
        P[:, start:end] = S_block_exp  # Store unnormalized probabilities

    # Step 4: Normalize by accumulated denominator
    P /= row_sum_exp[:, None]  # Row-wise normalization

    return P


# Example Usage
L = 8  # Example sequence length (use larger values like 4096 for real cases)
block_size = 4  # Block size
S = np.random.randn(L, L)  # Random attention scores

P_normal = [softmax(x) for x in S]
print("\nSoftmax directly:\n", pd.DataFrame(P_normal))

P_block = block_softmax(S, block_size)
print("\nSoftmax output (block-wise computed):\n", pd.DataFrame(P_block))
print("\nRow sums (should be ~1):\n", P_block.sum(axis=1))
