import torch
import math

def benchmark_corope_backward_only(
    B: int = 1,
    H: int = 16,
    seq_len: int = 4096,
    D: int = 128,
    q_dtype: torch.dtype = torch.float16,
    k_dtype: torch.dtype = torch.float16,
    v_dtype: torch.dtype = torch.float16,
):
    from CoROPE.co_rope_backward import co_rope_backward

    device = "cuda"
    torch.manual_seed(0)

    NUM_WARMUP = 50
    NUM_REPS = 100

    print(f"Benchmarking CoRoPE Backward Only with: B={B}, H={H}, D={D}, seq_len={seq_len}")

    # Create dummy forward outputs (simulating pre-computed forward pass)
    q_out = torch.randn(B, H, seq_len, D, device=device, dtype=q_dtype)
    k_out = torch.randn(B, H, seq_len, D, device=device, dtype=k_dtype)
    v_out = torch.randn(B, H, seq_len, D, device=device, dtype=v_dtype)
    o_out = torch.randn(B, H, seq_len, D, device=device, dtype=q_dtype)
    m_out = torch.randn(B, H, seq_len, device=device, dtype=torch.float32)
    l_out = torch.randn(B, H, seq_len, device=device, dtype=torch.float32)
    a_k_out = torch.randn(B, H, seq_len, device=device, dtype=torch.float32)
    do = torch.randn(B, H, seq_len, D, device=device, dtype=q_dtype)

    sm_scale = 1.0 / math.sqrt(D)

    # Prepare thetas
    theta = 10000.0
    thetas = 1.0 / (
        theta ** (torch.arange(0, D, 2, device=device).float() / D)
    )
    thetas = thetas.repeat_interleave(2)
    thetas = thetas.unsqueeze(0).repeat(H, 1).contiguous()

    # Warmup
    print("Warming up backward pass...")
    for _ in range(NUM_WARMUP):
        dq, dk, dv, dthetas = co_rope_backward(
            q_out, k_out, v_out, o_out, m_out, l_out, a_k_out, do, thetas, True, sm_scale
        )
    torch.cuda.synchronize()

    # Benchmark Backward
    print("Benchmarking backward pass...")
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()
    start_event.record(stream)
    for _ in range(NUM_REPS):
        dq, dk, dv, dthetas = co_rope_backward(
            q_out, k_out, v_out, o_out, m_out, l_out, a_k_out, do, thetas, True, sm_scale
        )
    end_event.record(stream)
    torch.cuda.synchronize()
    ms_bwd = start_event.elapsed_time(end_event) / NUM_REPS

    # FLOPs estimation for backward
    flops_bwd = B * H * seq_len * seq_len * (6 * D)  # Rough estimate
    tflops_bwd = flops_bwd / (ms_bwd * 1e-3) / 1e12

    # Print results
    line = "=" * 80
    print(line)
    print(f"Backward Only Benchmark Results:")
    print(f"Config: B={B}, H={H}, S={seq_len}, D={D}")
    print(f"Backward FLOPs: {flops_bwd/1e12:.2f} TF")
    print(line)
    print(f"{'Operation':<20} {'Latency (ms)':>15}   {'TFLOP/s':>9}")
    print("-" * 80)
    print(f"{'CoRoPE Backward':<20} {ms_bwd:15.3f}   {tflops_bwd:9.2f}")
    print(line)

def benchmark_corope_fwd_flops(
    B: int = 1,
    H: int = 16,
    seq_len: int = 4096,
    D: int = 128,
    q_dtype: torch.dtype = torch.float16,
    k_dtype: torch.dtype = torch.float16,
    v_dtype: torch.dtype = torch.float16,
):
    from CoROPE.co_rope_forward import co_rope_forward
    from CoROPE.co_rope_backward import co_rope_backward

    device = "cuda"
    torch.manual_seed(0)
    NUM_WARMUP = 50
    NUM_REPS = 100
    dtype_str = (
        f"q={str(q_dtype).split('.')[-1]}, "
        f"k={str(k_dtype).split('.')[-1]}, "
        f"v={str(v_dtype).split('.')[-1]}"
    )

    print(
        f"Benchmarking CoRoPE with: B={B}, H={H}, D={D}, {dtype_str}, "
        f"Warmup={NUM_WARMUP}, Reps={NUM_REPS}"
    )

    S = seq_len
    print("\n" + "-" * 5 + f" Sequence Length (S) = {S} " + "-" * 5)

    q_val = torch.randn(B, H, S, D, device=device, dtype=q_dtype)
    k_val = torch.randn_like(q_val, dtype=k_dtype)
    v_val = torch.randn_like(q_val, dtype=v_dtype)

    sm_scale = 1.0 / math.sqrt(D)

    q_f, k_f, v_f = q_val.clone(), k_val.clone(), v_val.clone()

    a_k_out = torch.empty((B, H, seq_len), device=device, dtype=torch.float32)

    # Prepare thetas for backward pass
    theta = 10000.0
    thetas = 1.0 / (
        theta ** (torch.arange(0, D, 2, device=device).float() / D)
    )
    thetas = thetas.repeat_interleave(2)
    thetas = thetas.unsqueeze(0).repeat(H, 1).contiguous()

    # Warmup
    print("Warming up forward pass...")
    for _ in range(NUM_WARMUP):
        (q_out, k_out, v_out, o_out, m_out, l_out, a_k_out), output = co_rope_forward(
            q_f, k_f, v_f, causal=True, sm_scale=sm_scale, a_k_out=a_k_out
        )
    torch.cuda.synchronize()

    print("Warming up backward pass...")
    do = torch.randn_like(output)
    for _ in range(NUM_WARMUP):
        dq, dk, dv, dthetas = co_rope_backward(
            q_out, k_out, v_out, o_out, m_out, l_out, a_k_out, do, thetas, True, sm_scale
        )
    torch.cuda.synchronize()

    # Benchmark Forward
    print("Benchmarking forward pass...")
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()
    start_event.record(stream)
    for _ in range(NUM_REPS):
        (q_out, k_out, v_out, o_out, m_out, l_out, a_k_out), output = co_rope_forward(
            q_f, k_f, v_f, causal=True, sm_scale=sm_scale, a_k_out=a_k_out
        )
    end_event.record(stream)
    torch.cuda.synchronize()
    ms_fwd = start_event.elapsed_time(end_event) / NUM_REPS

    # Benchmark Backward
    print("Benchmarking backward pass...")
    do = torch.randn_like(output)
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record(stream)
    for _ in range(NUM_REPS):
        dq, dk, dv, dthetas = co_rope_backward(
            q_out, k_out, v_out, o_out, m_out, l_out, a_k_out, do, thetas, True, sm_scale
        )
    end_event.record(stream)
    torch.cuda.synchronize()
    ms_bwd = start_event.elapsed_time(end_event) / NUM_REPS

    # FLOPs estimation
    SKV = S
    flops_fwd = B * H * S * SKV * (4 * D)
    flops_bwd = B * H * S * SKV * (6 * D)  # Backward typically has more operations
    tflops_fwd = flops_fwd / (ms_fwd * 1e-3) / 1e12
    tflops_bwd = flops_bwd / (ms_bwd * 1e-3) / 1e12
    tflops_total = (flops_fwd + flops_bwd) / ((ms_fwd + ms_bwd) * 1e-3) / 1e12

    # Print results
    line = "=" * 80
    print(line)
    print(
        f"Config: B={B}, H={H}, S={S}, D={D}, "
        f"q_dtype={q_dtype}, k_dtype={k_dtype}, v_dtype={v_dtype}"
    )
    print(f"Forward FLOPs: {flops_fwd/1e12:.2f} TF, Backward FLOPs: {flops_bwd/1e12:.2f} TF")
    print(line)
    print(f"{'Operation':<20} {'Latency (ms)':>15}   {'TFLOP/s':>9}")
    print("-" * 80)
    print(f"{'CoRoPE Forward':<20} {ms_fwd:15.3f}   {tflops_fwd:9.2f}")
    print(f"{'CoRoPE Backward':<20} {ms_bwd:15.3f}   {tflops_bwd:9.2f}")
    print(f"{'Total':<20} {ms_fwd + ms_bwd:15.3f}   {tflops_total:9.2f}")
    print(line)

    # Memory usage estimation
    memory_fwd = (B * H * S * D * 4) / 1e9  # GB, assuming float16
    memory_bwd = (B * H * S * D * 8) / 1e9  # GB, including gradients
    print(f"Estimated Memory Usage:")
    print(f"  Forward: {memory_fwd:.2f} GB")
    print(f"  Backward: {memory_bwd:.2f} GB")
    print(f"  Total: {memory_fwd + memory_bwd:.2f} GB")

if __name__ == "__main__":
    print("Running CoRoPE Forward + Backward Benchmark")
    benchmark_corope_fwd_flops()

 
 