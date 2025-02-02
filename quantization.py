import numpy as np


# x_quantized =round(x / delta)
# delta is a scaling factor that determins how floating points are mappedto lower-bit values

# float32, 32-bit [1 sign, 8 exponent, 23 mantissa], memory 4 bytes
# uint8， only 8 bits -> 1 byte


def quantize_4bit(tensor):
    """Quantizes a FP32 tensor into 4-bit integers."""
    # Step 1: Find min/max for scaling
    min_val, max_val = tensor.min(), tensor.max()

    # Step 2: Compute scale and zero-point
    scale = (max_val - min_val) / (2**4 - 1)  # 16 levels (4-bit)
    quantized = np.round((tensor - min_val) / scale).astype(
        np.uint8
    )  # Map to 4-bit values

    return quantized, scale, min_val  # Store scale & min_val for later dequantization


def quantized_matmul(A, B, scale_A, scale_B):
    """Performs approximate matrix multiplication on quantized values."""
    # Quantized matrices (int4 values)
    A_q, scale_A, min_A = quantize_4bit(A)
    B_q, scale_B, min_B = quantize_4bit(B)

    # Integer matrix multiplication (fast on hardware)
    int_matmul = np.dot(A_q.astype(np.float32), B_q.astype(np.float32))

    # Restore to FP32 using scale factors
    return int_matmul * (scale_A * scale_B)


def dequantize_4bit(q_tensor, scale, min_val):
    """Dequantizes a 4-bit tensor back to FP32."""
    return q_tensor * scale + min_val  # Reverse the quantization process


# Example usage
fp32_tensor = np.random.randn(8, 8).astype(
    np.float32
)  # 8x8 random FP32 tensor, 8 * 8 * 4 bytes => 256 bytes
print("\nOriginal Tensor:\n", fp32_tensor)
print("\nOriginal Tensor memory usage (FP32):", fp32_tensor.nbytes, "bytes")

q_tensor, scale, min_val = quantize_4bit(fp32_tensor)  # 8 * 8 * 1 byte => 64 bytes
print("\nQuantized (4-bit) Tensor:\n", q_tensor)
print("\nQuantized (4-bit) Tensor memory usage (FP32):", q_tensor.nbytes, "bytes")
print("Scale factor:", scale)
print("Min value:", min_val)

# Restore original values
restored_tensor = dequantize_4bit(q_tensor, scale, min_val)
print("\nRestored FP32 Tensor:\n", restored_tensor)


# Optimized Matrix Multiplication with Quantized Tensors
# Example Usage
A = np.random.randn(8, 8).astype(np.float32)
B = np.random.randn(8, 8).astype(np.float32)
n_resutl = A @ B
q_result = quantized_matmul(A, B, 0.1, 0.2)

print("\nNormal MatMul Result:\n", n_resutl)
print("\nQuantized MatMul Result:\n", q_result)
