import numpy as np
from scipy.special import softmax
import pandas as pd

# Limits decimals, removes scientific notation
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={"float": lambda x: f"{x:04.2f}"})
pd.options.display.float_format = "{:04.5f}".format


def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    # Attention scores represents the similarity between Q and K (how relevant is a token?)
    attn_scores = Q @ K.T / np.sqrt(d_k)  # Scaled dot product
    # Attention weights,
    # Softmax-normalized scores (focus level on each token)
    attn_weights = softmax(attn_scores, axis=-1)
    return attn_weights @ V, attn_weights


def multi_head_attention(Q, K, V, num_heads):
    d_k = Q.shape[-1]
    d_v = V.shape[-1]

    # Step 1: Split Q, K, V into multiple heads
    Q_split = np.split(Q, num_heads, axis=-1)
    K_split = np.split(K, num_heads, axis=-1)
    V_split = np.split(V, num_heads, axis=-1)

    # Step 2: Apply scaled dot-product attention for each head
    outputs = []
    for i in range(num_heads):
        output, _ = scaled_dot_product_attention(Q_split[i], K_split[i], V_split[i])
        outputs.append(output)

    # Step 3: Concatenate outputs from all heads
    concat_output = np.concatenate(outputs, axis=-1)

    # Step 4: Apply final linear transformation (dummy for simplicity, no learned weights)
    final_output = (
        concat_output  # In practice, this would be a learned linear transformation.
    )

    return final_output


if __name__ == "__main__":
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    X = np.array(
        [
            np.array([0.1, 0.2, 0.3, 0.4]),  # the
            np.array([0.5, 0.6, 0.7, 0.8]),  # cat
            np.array([0.9, 1.0, 1.1, 1.2]),  # sat
            np.array([1.3, 1.4, 1.5, 1.6]),  # on
            np.array([0.2, 0.3, 0.3, 0.4]),  # the
            np.array([0.6, 0.7, 0.8, 0.9]),  # mat
        ],
    )
    print("Input:", X.shape)  # (6, 4)
    print("\nInput:\n", pd.DataFrame(X))

    # **Self-Attention**

    # Init W_Q, W_K, W_V, their weight should be learned during training.
    d_k = X.shape[-1]  # dims 6
    W_Q = np.random.rand(d_k, d_k)
    W_K = np.random.rand(d_k, d_k)
    W_V = np.random.rand(d_k, d_k)

    # Calculate Q, K, V
    Q = X @ W_Q  # (384,)  What information should I focus on?
    K = X @ W_K  # (384,)  What does this token represent?
    V = X @ W_V  # (384,)  What information does this token contain?

    # Attention scores represents the similarity between Q and K (how relevant is a token?)
    scores = Q @ K.T / np.sqrt(d_k)
    print("\nAttention Score:\n", pd.DataFrame(scores))
    # Attention weights,
    # Softmax-normalized scores (focus level on each token)
    attention_weights = [softmax(x) for x in scores]
    print("\nAttention Weights:\n", pd.DataFrame(attention_weights))

    # Output
    weighted_values = attention_weights @ V
    print("\Weighted Values:\n", pd.DataFrame(weighted_values))

    # Displaying the attention weights
    print("\nOriginal sentence:\n", tokens)
    for i, word in enumerate(tokens):
        print("\nFor the word: ", word)
        top_3 = sorted(
            range(len(tokens)), key=lambda j: attention_weights[i][j], reverse=True
        )[:3]
        print(top_3)
        for j in top_3:
            print(f"Attention for: {tokens[j]}: {attention_weights[i][j]:.4f}")

    # Explanation of attention relationships
    print("\nExplanation:")
    for i, word in enumerate(tokens):
        sorted_indices = np.argsort(-attention_weights[i])
        second_max_attention = sorted_indices[1]  # Ignoring the first (self-attention)
        print(f"The word '{word}' is most attended by '{tokens[second_max_attention]}'")
