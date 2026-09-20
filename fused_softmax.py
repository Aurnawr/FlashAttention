import torch 
import triton 
import triton.language as tl

device = torch.device(f'cuda:{torch.cuda.current_device()}')

# step 1
def naive_softmax(x):
    # assume x shape (M,N)---> here we are summing across the columns, also x resides in the DRAM

    # read MN elements and write M elements
    x_max = x.max (dim=1)[0] # shape (M)

    z = x - x_max[:,None] # read MN + M elements, MN flops, then write MN elements 
    numerator = torch.exp(z) # read MN elements, write MN elements
    denominator = numerator.sum(dim=1) # read MN elements, write M elements

    out = numerator/ denominator[:, None] # read MN + M elements, write MN

    return out # in total, we did 8MN + 4M memory operations


# step 3 
properties = triton.runtime.driver.active.utils.get_device_properties(device.index)
num_sm = properties["multiprocessor_count"]
num_regs = properties['max_num_regs']
sram_per_sm = properties['max_shared_mem']
warp_size = properties['warpSize']



def softmax (x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE>= 2048:
        num_warps = 8 
    if BLOCK_SIZE>= 4096:
        num_warps = 16

    # pipelining (loading memory while the gpu is executing)
    num_stages = 4 if sram_per_sm>=200_000 else 2 
    y = torch.empty_like(x)
    kernel =_softmax_kernel.warmup(
        x,y,
        x.stride(0), y.stride(0),
        n_rows,n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=(1,)
    )
    kernel._init_handles()
    n_regs_per_program = kernel.n_regs # ------> amount of registers in each thread residing in that program 
    sram_needed_per_program = kernel.metadata.shared
    reg_occupancy = num_regs // (n_regs_per_program * warp_size * num_warps)
    # num_regs = 65536 ----> regs in an SM
    # each program might use 
        # n_regs_per_program = 32
        # n_warps = 8
        # warp_size = 32
    # each program needs (n_regs_per_program * warp_size * num_warps) registers 
    # 65536 // ( 8 * 32 * 32 ) = 8 programs per SM
    sram_occupancy = sram_per_sm // sram_needed_per_program
    programs_per_sm = min( reg_occupancy, sram_occupancy )
    num_programs = min (num_sm * programs_per_sm, n_rows)
    grid = (num_programs, 1, 1)
    kernel[grid](
        x,y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,

    )
    return y 
# basically, in the above function, we try to heuristically define the hyperparameters such as grid on the basis of block size. 

# step 4
@triton.jit 
def _softmax_kernel(
    input_ptr, output_ptr, 
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr, # -------> we dont have to pass because it already stored during warmup calling
    num_stages: tl.constexpr,
):
    # each PID handles one row 
    PID = tl.program_id(0)
    row_step = tl.num_programs(0) # -----> we will use this only when we have a remainder i.e n_rows > num_programs 

    #calculating the softmax for each single row in this loop   -----> only one read and write per row so total read and writes = 2M
    for row_idx in tl.range(PID, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0,BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets #---> all the pointers from the start of the row till the end (till the blocksize)
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf') ) # shape (BLOCK_SIZE) which is roughly (n_cols), also the places where mask is applied become other i.e -infinity

        # fused the kernel 
        row_minus_max = row - tl.max(row,axis = 0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator /denominator

        output_row_start_pointer = output_ptr + row_idx * output_row_stride


        tl.store(output_row_start_pointer + col_offsets, softmax_output, mask = mask)


# step 2 
def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=device):
    torch.manual_seed(0)
    x=torch.randn(size[0],size[1],device=device)
    z_tri = softmax(x)
    z_torch = torch.softmax(x,axis = 1)
    torch.testing.assert_close(z_tri, z_torch, atol=atol, rtol=rtol)
    print ("PASSED")

#step 5
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096} # values for function arguments not in x_names
    ))
def benchmark(M, N, provider):
    # making the input data
    x = torch.randn(M, N, device=device, dtype=torch.float32)

    # these two lines ensure more accurate benchmarks; i usually forget to use them but it's not a big deal
    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        # 2 = number of memory operations (1 read + 1 write)
        # x.numel() = number of elements
        # x.element_size() = bytes per element (4 for float32)
        # 1e-9 converts bytes to GB
        # 1e-3 converts milliseconds to seconds
    return gbps(ms)

if __name__ == "__main__":
    # always run unit-tests
    test_softmax_kernel(size=(1823, 781))

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)