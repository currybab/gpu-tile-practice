import cuda.tile as ct
import cupy
import math

@ct.kernel
def gelu_kernel(input, output, n: int, m: int, N_TILE: ct.Constant[int], M_TILE: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    input_tile = ct.load(input, index=(bidx, bidy), shape=(N_TILE, M_TILE))
    output_tile = 0.5 * input_tile * (1 + ct.tanh(ct.sqrt(2 / math.pi) * (input_tile + 0.044715 * input_tile ** 3)))
    ct.store(output, index=(bidx, bidy), tile=output_tile)

    

# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: input, output are all float32 device tensors
def solution(input, output, n: int, m: int):
    N_TILE = 32
    M_TILE = 64
    grid = (ct.cdiv(n, N_TILE), ct.cdiv(m, M_TILE))
    ct.launch(cupy.cuda.get_current_stream(), grid, gelu_kernel, (input, output, n, m, N_TILE, M_TILE))


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    # Test with 6144x4096 tensor
    n, m = 6144, 4096

    # Create random input tensor
    input_torch = torch.randn(n, m, dtype=torch.float32, device="cuda")
    output_torch = torch.zeros(n, m, dtype=torch.float32, device="cuda")

    # Convert to cupy for cuda.tile
    input_cupy = cupy.asarray(input_torch)
    output_cupy = cupy.asarray(output_torch)

    # Run cuda.tile GELU
    solution(input_cupy, output_cupy, n, m)

    # PyTorch reference (approximate GELU)
    expected = F.gelu(input_torch, approximate='tanh')

    # Convert result back to torch for comparison
    result = torch.as_tensor(output_cupy, device="cuda")

    # Check correctness
    if torch.allclose(result, expected, rtol=1e-4, atol=1e-5):
        print(f"✓ GELU test passed! Shape: ({n}, {m})")
    else:
        diff = torch.abs(result - expected).max().item()
        mean_diff = torch.abs(result - expected).mean().item()
        print(f"✗ GELU test failed!")
        print(f"  Max diff: {diff}")
        print(f"  Mean diff: {mean_diff}")
