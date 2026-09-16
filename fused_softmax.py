import torch 
import triton 
import triton.language as tl

device = torch.device(f'cuda:{torch.cuda.current_device()}')

# step 1
def naive_softmax(x):
    # assume x shape (M,N)---> here we are summing across the columns 

    # read MN elements and write M elements
    x_max = x.max (dim=1)[0] # shape (M)

    z = x - x_max[:,None] # read MN + M elements, MN flops, then write MN elements 
    numerator = torch.exp(z) # read MN elements, write MN elements
    denominator = numerator.sum(dim=1) # read MN elements, write M elements

    out = numerator/ denominator[:, None] # read MN + M elements, write MN

    return out # in total, we did 8MN + 4M memory operations
