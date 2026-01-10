import cuda.tile as ct
import cupy

@ct.kernel
def conv1d_kernel(A, B, C, N: int, K: ct.Constant[int], TILE: ct.Constant[int]):
    bidx = ct.bid(0)
    out_indices = bidx * TILE + ct.arange(TILE, dtype=ct.int32)

    acc = ct.zeros((TILE,), dtype=ct.float32)

    for k in range(K):
        A_load = ct.gather(A, out_indices + k - K // 2)
        B_load = ct.gather(B, k)
        acc = acc + A_load * B_load

    ct.store(C, index=(bidx,), tile=acc)


# Input
# - Vector A of size N (input signal)
# - Vector B of size K (filter)
# Output
# - Vector C of size N (convolved signal)
# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: A, B, C are all float32 device tensors
def solution(A, B, C, N: int, K: int):
    TILE = 256
    grid = (ct.cdiv(N, TILE),)
    ct.launch(cupy.cuda.get_current_stream(), grid, conv1d_kernel, (A, B, C, N, K, TILE))


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    # Test parameters
    N = 131072  # Signal length
    K = 3       # Filter size

    # Create random input signal and filter
    A_torch = torch.randn(N, dtype=torch.float32, device="cuda")
    B_torch = torch.randn(K, dtype=torch.float32, device="cuda")
    C_torch = torch.zeros(N, dtype=torch.float32, device="cuda")

    # Convert to cupy for cuda.tile
    A_cupy = cupy.asarray(A_torch)
    B_cupy = cupy.asarray(B_torch)
    C_cupy = cupy.asarray(C_torch)

    # Run cuda.tile conv1d
    solution(A_cupy, B_cupy, C_cupy, N, K)

    # PyTorch reference - using F.conv1d with centered padding
    # C[i] = A[i-K//2]*B[0] + A[i-K//2+1]*B[1] + ... + A[i-K//2+K-1]*B[K-1]
    A_reshaped = A_torch.view(1, 1, N)
    B_reshaped = B_torch.view(1, 1, K)
    expected = F.conv1d(A_reshaped, B_reshaped, padding=K // 2).view(N)

    # Convert result back to torch for comparison
    result = torch.as_tensor(C_cupy, device="cuda")

    # Check correctness
    if torch.allclose(result, expected, rtol=1e-4, atol=1e-4):
        print(f"✓ Conv1D test passed! N={N}, K={K}")
    else:
        diff = torch.abs(result - expected).max().item()
        print(f"✗ Conv1D test failed! Max diff: {diff}")
