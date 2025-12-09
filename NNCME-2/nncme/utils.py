"""Utility helpers for logging, checkpointing, and tensor conversions."""

import os
from glob import glob

import numpy as np
import torch

from nncme.args import args
if args.dtype == 'float32':
    default_dtype = np.float32
    default_dtype_torch = torch.float32
elif args.dtype == 'float64':
    default_dtype = np.float64
    default_dtype_torch = torch.float64
else:
    raise ValueError('Unknown dtype: {}'.format(args.dtype))

np.seterr(all='raise')
np.seterr(under='warn')
np.set_printoptions(precision=8, linewidth=160)

torch.set_default_dtype(default_dtype_torch)
torch.set_printoptions(precision=8, linewidth=160)
torch.backends.cudnn.benchmark = True

if not args.seed:
    args.seed = np.random.randint(1, 10**8)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
args.device = torch.device('cpu' if args.cuda < 0 else 'cuda:0')

args.out_filename = None


def get_ham_args_features():
    """Return run-identifying strings used to build output paths."""

    model = '{Model}'.format(**vars(args))
    ham_args = '{Model}_L{L}_S{Sites}_M{M}_T{Tstep}_dt{delta_t}_batch{batch_size}'
    ham_args = ham_args.format(**vars(args))

    if args.net == 'made':
        features = 'nd{net_depth}_nw{net_width}_made'
    elif args.net == 'bernoulli':
        features = 'bernoulli_nw{net_width}'
    elif args.net == 'rnn':
        features = 'nd{net_depth}_nw{net_width}_rnn'
    elif args.net == 'transformer':
        features = 'dm{d_model}_df{d_ff}_ly{n_layers}_he{n_heads}_trans'
    elif args.net == 'rnnNo':
        features = 'nd{net_depth}_nw{net_width}_rnnNo'
    elif args.net == 'rnn2':
        features = 'nd{net_depth}_nw{net_width}_rnn2'
    elif args.net == 'rnn3':
        features = 'nd{net_depth}_nw{net_width}_rnn3'
    elif args.net == 'pixelcnn':
        features = 'nd{net_depth}_nw{net_width}_pixelcnn'
    elif args.net == 'NADE':
        features = 'nd{net_depth}_nw{net_width}_NADE'
    else:
        features = 'nd{net_depth}_nw{net_width}_hks{half_kernel_size}'
    
    features += '_{method}'
    # features += '_L{L}'
    features += '_lr{lr}'
    features += '_epoch{epoch}'
    features += '_Loss{lossType}'
    features += '_Sampling{sampling}'
    if args.sampling =='manual' or args.sampling == 'random':
        features += '{ESNumber}'
    if args.sampling =='diffusive':
        features += '_kernel{kernel}'
    if args.sampling =='power':
        features += '_alpha{alpha}'
    if args.absorbed==True:
        features += '_absorbed{absorb_state}'
    if args.modify:
        features += '_modify'
    # features += '_Order{order}'
    features += '_IniDist{IniDistri}'
    features += '_Para{Para}'
    
    # if args.lossType=='kl':
    #     features += '_Losskl'
    # elif args.lossType=='l2':
    #     features += '_Lossl2'
    # elif args.lossType=='he':
    #     features += '_Losshe'           

    if args.bias:
        features += '_bias'
    if args.AdaptiveT:
        features += 'AdaptiveT{AdaptiveTFold}'
    if args.binary:
        features += '_binary'
    if args.conservation>1:
        features += '_conser{conservation}'
    if args.reverse:
        features += '_WithRev'
    # if args.z2:
    #     features += '_z2'
    if args.res_block:
        features += '_res'
    # if args.x_hat_clip:
    #     features += '_xhc{x_hat_clip:g}'
    # if args.final_conv:
    #     features += '_fconv'

    if args.optimizer != 'adam':
        features += '_{optimizer}'
    if args.lr_schedule:
        features += '_lrs'
    if args.beta_anneal:
        features += '_ba{beta_anneal:g}'
    if args.clip_grad:
        features += '_cg{clip_grad:g}'

    features = features.format(**vars(args))

    return model, ham_args, features


def init_out_filename():
    """Populate ``args.out_filename`` with a unique output directory path."""

    if not args.out_dir:
        return
    model, ham_args, features = get_ham_args_features()

    template = '{args.out_dir}/{model}/{ham_args}/{features}'#/out{args.out_infix}'
    args.out_filename = template.format(**{**globals(), **locals()})
    # print(args.out_filename)
    # print(os.path.exists(args.out_filename))
    counter = 1
    while os.path.exists(args.out_filename):
        template = '{args.out_dir}/{model}/{ham_args}/({counter}) {features}'#/out{args.out_infix}'
        args.out_filename = template.format(**{**globals(), **locals()})
        counter += 1
        
    args.out_filename=args.out_filename+'/out{}'.format(args.out_infix)
    # print(args.out_filename)
    
def ensure_dir(filename):
    """Create parent directories for ``filename`` when they do not exist."""

    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    """Prepare output directories and checkpoint folders for the run."""

    if not args.out_dir:
        return
    init_out_filename()
    ensure_dir(args.out_filename)
    if args.save_step:
        ensure_dir(args.out_filename + '_save/')
    if args.visual_step:
        ensure_dir(args.out_filename + '_img/')
    ensure_dir(args.out_filename + '_img/Data/')


def clear_log():
    """Truncate the log file for the active run."""

    if args.out_filename:
        open(args.out_filename + '.log', 'w').close()


def clear_err():
    """Truncate the error log file for the active run."""

    if args.out_filename:
        open(args.out_filename + '.err', 'w').close()


def my_log(s):
    """Write ``s`` both to stdout and to the persistent run log."""

    if args.out_filename:
        with open(args.out_filename + '.log', 'a', newline='\n') as f:
            f.write(s + '\n')
    if not args.no_stdout:
        print(s)


def my_err(s):
    """Write ``s`` to the error log and mirror it to stdout."""

    if args.out_filename:
        with open(args.out_filename + '.err', 'a', newline='\n') as f:
            f.write(s + '\n')
    if not args.no_stdout:
        print(s)


def print_args(print_fn=my_log):
    """Print args operation.

    

    Args:

        print_fn: Parameter forwarded to print_args.

    

    Returns:

        Result produced by print_args.

    """


    for k, v in args._get_kwargs():
        print_fn('{} = {}'.format(k, v))
    print_fn('')


def parse_checkpoint_name(filename):
    """Parse checkpoint name operation.

    

    Args:

        filename: Parameter forwarded to parse_checkpoint_name.

    

    Returns:

        Result produced by parse_checkpoint_name.

    """


    filename = os.path.basename(filename)
    filename = filename.replace('.state', '')
    step = int(filename)
    return step


def get_last_checkpoint_step():
    """Get last checkpoint step operation.

    

    Returns:

        Result produced by get_last_checkpoint_step.

    """


    if not (args.out_filename and args.save_step):
        return -1
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    if not filename_list:
        return -1
    step = max([parse_checkpoint_name(x) for x in filename_list])
    return step


def clear_checkpoint():
    """Clear checkpoint operation.

    

    Returns:

        Result produced by clear_checkpoint.

    """


    if not (args.out_filename and args.save_step):
        return
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    for filename in filename_list:
        os.remove(filename)


# Do not load some params
def ignore_param(state, net):
    """Ignore param operation.

    

    Args:

        state: Parameter forwarded to ignore_param.

        net: Parameter forwarded to ignore_param.

    

    Returns:

        Result produced by ignore_param.

    """


    ignore_param_name_list = ['x_hat_mask', 'x_hat_bias']
    param_name_list = list(state.keys())
    for x in param_name_list:
        for y in ignore_param_name_list:
            if y in x:
                state[x] = net.state_dict()[x]
                break

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def dec2bin(x, length):
    """Dec2bin operation.

    

    Args:

        x: Parameter forwarded to dec2bin.

        length: Parameter forwarded to dec2bin.

    

    Returns:

        Result produced by dec2bin.

    """


    mask = 2 ** torch.arange(length - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).int()


def bin2dec(b, length):
    """Bin2dec operation.

    

    Args:

        b: Parameter forwarded to bin2dec.

        length: Parameter forwarded to bin2dec.

    

    Returns:

        Result produced by bin2dec.

    """


    mask = 2 ** torch.arange(length - 1, -1, -1).to(b.device, torch.int)
    return torch.sum(mask * b.int(), -1)


def gen_all_binary_vectors(length: int) -> torch.Tensor:
    """Gen all binary vectors operation.

    

    Args:

        length: Parameter forwarded to gen_all_binary_vectors.

    

    Returns:

        Result produced by gen_all_binary_vectors.

    """


    return (torch.arange(2**length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
# this function is the same as F.scaled_dot_product_attention
# but is more efficient for per-sample gradients
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """Scaled dot product attention operation.

    

    Args:

        query: Parameter forwarded to scaled_dot_product_attention.

        key: Parameter forwarded to scaled_dot_product_attention.

        value: Parameter forwarded to scaled_dot_product_attention.

        attn_mask: Parameter forwarded to scaled_dot_product_attention.

        dropout_p: Parameter forwarded to scaled_dot_product_attention.

        is_causal: Parameter forwarded to scaled_dot_product_attention.

        scale: Parameter forwarded to scaled_dot_product_attention.

    

    Returns:

        Result produced by scaled_dot_product_attention.

    """


    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if is_causal:
        causal_mask = torch.tril(torch.ones(L, S, dtype=torch.bool, device=query.device))
        attn_bias = attn_bias.masked_fill(~causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return torch.matmul(attn_weight, value)


def cholesky_solve(O_mat, F_vec, lambd=1e-3):
    """
    Solve the linear system `(O^T O + lambda I) dtheta = O^T R = F` by Cholesky decomposition

    see arXiv:2310.17556, Algorithm 1
    """
    N, _ = O_mat.size()
    W = O_mat @ O_mat.T + lambd * torch.eye(N, device=O_mat.device)
    L = torch.linalg.cholesky(W)
    Q = torch.linalg.inv(L) @ O_mat

    return (F_vec - Q.T @ Q @ F_vec) / lambd


def cholesky_solve_fast(O_mat, F_vec, lambd=1e-3):
    """
    Solve the linear system `(O^T O + lambda I) dtheta = O^T R = F` by Cholesky decomposition

    The computation Q is inlined

    see arXiv:2310.17556, Algorithm 1
    """
    N, _ = O_mat.size()
    W = O_mat @ O_mat.T + lambd * torch.eye(N, device=O_mat.device)
    L = torch.linalg.cholesky(W)
    QTQF = O_mat.T @ torch.cholesky_solve(O_mat, L) @ F_vec

    return (F_vec - QTQF) / lambd


def svd_solve(O_mat, F_vec, lambd=1e-3):
    """
    Solve the linear system `(O^T O + lambda I) dtheta = F` by svd

    First compute `O O^T = U @ Sigma^2 @ U^T`, then `V = O^T U Sigma^{-1}`
    
    see arXiv:2310.17556, Appendix C
    """
    Sigma2, U = torch.linalg.eigh(O_mat @ O_mat.T)
    V = O_mat.T @ (1.0 / torch.sqrt(Sigma2) * U)  # V = O^T U Sigma^{-1}

    return (V * (1.0 / (Sigma2 + lambd))) @ V.T @ F_vec + (F_vec - V @ V.T @ F_vec) / lambd


def minsr_solve(O_mat, R_vec, lambd=1e-3, r_pinv=1e-12, a_pinv=0, soft=True):
    """
    Solve the linear system `(O^T O + lambda I) dtheta = O^T R` by minSR

    `dtheta = (O^T O + lambda I)^-1 O^T R = O^T (O O^T + lambda I)^-1 R`

    Compute `O O^T = U @ D @ U^T`, so `(O O^T)^-1 = U @ D^-1 @ U^T`

    see arXiv:2108.03409 Section 2.5, and arXiv:2302.01941 MinSR solution
    """
    N, _ = O_mat.size()
    D, U = torch.linalg.eigh(O_mat @ O_mat.T + lambd * torch.eye(N, device=O_mat.device))
    threshold = D.max().abs() * r_pinv + a_pinv
    if soft:
        D_inv = 1 / (D * (1 + (threshold / torch.abs(D)) ** 6))
    else:
        D_inv = torch.where(torch.abs(D) >= threshold, 1 / D, torch.tensor(0.0, device=O_mat.device, dtype=O_mat.dtype))
    T_inv = (U * D_inv) @ U.T

    return O_mat.T @ T_inv @ R_vec

def factorial(n):
    """Factorial operation.

    

    Args:

        n: Parameter forwarded to factorial.

    

    Returns:

        Result produced by factorial.

    """


    if n == 0 or n == 1:
        return 1
    else:
        return torch.prod(torch.arange(1, n + 1, dtype=torch.float32))
