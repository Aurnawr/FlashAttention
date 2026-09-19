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
properties = triton.runtime.driver.activate.utils.get_device_properties(device.index)
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

    kernel =__softmax_kernel.warmup(
        x,y,
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
    













# step 2 
def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=device):
    torch.manual_seed(0)
    x=torch.randn(size[0],size[1],device=device)
    z_tri = softmax(x)
    z_torch = torch.softmax(x,axis = 1)
    torch.testing.assert_close(z_tri, z_torch, atol=atol, rtol=rtol)
    print ("PASSED")

