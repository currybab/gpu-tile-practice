import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv1d_kernel(gX: cute.Tensor, gW: cute.Tensor, gY: cute.Tensor, stride: cutlass.Int32, padding: cutlass.Int32):
    # gX: Input tensor (L)
    # gW: Weight tensor (K)
    # gY: Output tensor (L + 2*P - K) // S + 1
    
    idx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    
    bdx, _, _ = cute.arch.block_dim()
    out_idx = bidx * bdx + idx
    
    if out_idx < cute.size(gY):
        acc = 0.0
        K = cute.size(gW)
        L = cute.size(gX)
        
        # input_idx = out_idx * stride - padding + k
        start_idx = out_idx * stride - padding
        
        for k in range(K):
            input_idx = start_idx + k
            if input_idx >= 0 and input_idx < L:
                acc += gX[input_idx] * gW[k]
        
        gY[out_idx] = acc

@cute.jit
def conv1d(X: cute.Tensor, W: cute.Tensor, Y: cute.Tensor, stride: cutlass.Int32, padding: cutlass.Int32):
    block_size = 128
    grid_size = (cute.size(Y) + block_size - 1) // block_size
    
    conv1d_kernel(X, W, Y, stride, padding).launch(
        grid=(grid_size, 1, 1),
        block=(block_size, 1, 1)
    )

def test_conv1d(L, K, S, P):
    print(f"Testing L={L}, K={K}, S={S}, P={P}")
    
    # Calculate output size: (L + 2*P - K) // S + 1
    out_size = (L + 2 * P - K) // S + 1
    
    x_torch = torch.randn(L, dtype=torch.float32, device="cuda")
    w_torch = torch.randn(K, dtype=torch.float32, device="cuda")
    y_torch = torch.zeros(out_size, dtype=torch.float32, device="cuda")
    
    # Compile and run
    conv1d_compiled = cute.compile(conv1d, from_dlpack(x_torch), from_dlpack(w_torch), from_dlpack(y_torch), cutlass.Int32(S), cutlass.Int32(P))
    conv1d_compiled(from_dlpack(x_torch), from_dlpack(w_torch), from_dlpack(y_torch), cutlass.Int32(S), cutlass.Int32(P))
    
    # Verification
    x_4d = x_torch.view(1, 1, L)
    w_4d = w_torch.view(1, 1, K)
    expected = torch.nn.functional.conv1d(x_4d, w_4d, stride=S, padding=P).view(-1)
    
    torch.cuda.synchronize()
    is_correct = torch.allclose(y_torch, expected, atol=1e-5)
    print(f"  Output size: {out_size}")
    print(f"  Verification: {'Success' if is_correct else 'Failure'}")
    if not is_correct:
        print(f"  Max diff: {(y_torch - expected).abs().max()}")
    return is_correct

if __name__ == "__main__":
    test_cases = [
        (1024, 3, 1, 0),
        (1024, 3, 1, 1), # Same padding for K=3
        (1024, 5, 1, 2), # Same padding for K=5
        (1024, 3, 2, 1),
        (2048, 7, 3, 3),
        (512, 11, 4, 5),
        (100, 3, 1, 0),
    ]
    
    all_success = True
    for L, K, S, P in test_cases:
        if not test_conv1d(L, K, S, P):
            all_success = False
    
    print("\nOverall Status:", "PASS" if all_success else "FAIL")
