import cuda.tile as ct
import cupy

EPSILON = 1e-5

@ct.kernel
def rms_norm_kernel(X, Y, B: int, N: int, B_TILE: ct.Constant[int], N_TILE: ct.Constant[int]):
    bid = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, N_TILE))
    sum_sq = ct.zeros((B_TILE,), dtype=ct.float32)
    for ni in range(num_tiles):
        X_tile = ct.load(X, index=(bid, ni), shape=(B_TILE, N_TILE))
        sum_sq += ct.sum(X_tile * X_tile, axis=1)

    norm = 1 / ct.sqrt(sum_sq / N + EPSILON)
    norm = ct.expand_dims(norm, 1)

    for ni in range(num_tiles):
        X_tile = ct.load(X, index=(bid, ni), shape=(B_TILE, N_TILE))
        Y_tile = X_tile * norm
        ct.store(Y, index=(bid, ni), tile=Y_tile)

# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: X, Y are all float32 device tensors
def solution(X, Y, B: int, N: int):
    B_TILE = 32
    N_TILE = 128
    grid = (ct.cdiv(B, B_TILE),)
    ct.launch(cupy.cuda.get_current_stream(), grid, rms_norm_kernel, (X, Y, B, N, B_TILE, N_TILE))


if __name__ == "__main__":
    import torch

    class RMSNormRef:
        def __init__(self, epsilon=1e-5):
            self.epsilon = epsilon

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=torch.float32):
                rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
                return x / rms

    test_configs = [
        (16, 128),
        (32, 256),
        (64, 512),
        (128, 1024),
        (256, 2048),
    ]

    print("Testing RMS Normalization:")
    all_passed = True
    ref = RMSNormRef(epsilon=EPSILON)

    for B, N in test_configs:
        X_torch = torch.randn(B, N, dtype=torch.float32, device="cuda")
        Y_torch = torch.zeros(B, N, dtype=torch.float32, device="cuda")

        X_cupy = cupy.asarray(X_torch)
        Y_cupy = cupy.asarray(Y_torch)

        solution(X_cupy, Y_cupy, B, N)

        expected = ref(X_torch)
        result = torch.as_tensor(Y_cupy, device="cuda")

        if torch.allclose(result, expected, rtol=1e-3, atol=1e-3):
            print(f"  ✓ B={B}, N={N}")
        else:
            diff = torch.abs(result - expected).max().item()
            print(f"  ✗ B={B}, N={N} - Max diff: {diff}")
            all_passed = False

    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
