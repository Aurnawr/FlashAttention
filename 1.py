import torch
import triton 
import triton.language as tl
DEVICE = torch.device (f'cuda: {torch.cuda.current_device()}')

# actual kernel 
@triton.jit # decorator tells that the function is a triton function
def add_kernel(
    x_ptr, # points to the first element of the array
    y_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr, # defines the argument as a constant during compile time 

):
    # “I am program PID, and I am responsible for this chunk of the array.”
    PID = tl.program_id(axis= 0) 
    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements # masking the elements which are out of bounds from the arrays 

    # “Load my chunk of X and Y, perform the operation on the whole chunk, and write my chunk of Z.”
    x = tl.load(x_ptr + offsets, mask = mask , other = None) 
    y = tl.load(y_ptr + offsets, mask = mask , other = None)
    z = x + y  

    #write data back to HBM
    tl.store(z_ptr + offsets, z , mask = None)



    

# add wrapper function 
def add (x,y):
    # preallocate empty tensor 
    z = torch.empty_like(x, device = DEVICE)

    # check tensors on same device 
    assert x.device == DEVICE and y.device == DEVICE

    # defining our launch grid
    n_elements = z.numel()
    grid = lambda meta: (triton.cdiv(n_elements,meta['BLOCK_SIZE']), ) # meta is a dictionary and grid is a tuple for number of programs  (4,)

    # calling the kernel 
    add_kernel[grid](    # launch add kernel over the whole grid 
        x,
        y,
        z,
        n_elements,
        BLOCK_SIZE= 1024 
    )

    return z






def test_add_kernel (size, atol=1e-3,rtol=1e-3,device = DEVICE):
    # create test data 
    torch.manual_seed(0)
    x = torch.randn(size, device = DEVICE)
    y = torch.rand(size, device= DEVICE)
    # run triton and pytorch kernels
    z_tri = add (x, y)
    z_ref = x + y 
    # compare 
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print ('pass')
