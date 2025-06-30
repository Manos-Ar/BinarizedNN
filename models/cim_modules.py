import sys
sys.path.append('/home/earapidis/Fast-Crossbar-Sim/python')
from crossbar import _task
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def checkerboard_last_cols(arr: np.ndarray, C: int) -> None:
    """
    Overwrite the last C columns of `arr` in-place with a checkerboard pattern of 0s and 1s.
    """
    n, m = arr.shape
    if C > m:
        raise ValueError("C cannot be larger than the number of columns m.")
    rows = np.arange(n)[:, None]
    cols = np.arange(m - C, m)
    pattern = (rows + cols) % 2
    arr[:, -C:] = pattern

def compliment(x):
    x = x.clone()
    neg = -1*x
    pos = x

    pos[pos==-1] = 0
    neg[neg==-1] = 0
    return pos, neg

####################################################

def _process_one_pixel(args):
    """
    Compute the conv2d_tiles result for a single (n,i,j) position.
    Returns (n, i, j, output_vector).
    """
    (n, i, j,
     input_vec,
     crossbar_weights,
     Cout,
     num_tiles_columns,
     Num_rows,
     Num_Columns,
     mode,
     checkboard) = args

    inp = input_vec[n, i, j, :, :]  # (num_tiles_rows, Num_rows)
    full_out = np.zeros(num_tiles_columns * Num_Columns, dtype=float)

    for col_idx in range(num_tiles_columns):
        weight_tiles = crossbar_weights[:, col_idx, :, :]  # (num_tiles_rows, Num_rows, Num_Columns)
        accum = np.zeros((weight_tiles.shape[0], Num_Columns), dtype=float)

        for t_idx, vec in enumerate(inp):
            W = weight_tiles[t_idx]
            if checkboard and col_idx == num_tiles_columns - 1:
                checkerboard_last_cols(W, Num_Columns - Cout)
            _, out_vec = _task((t_idx, vec), W, Num_rows, Num_Columns, mode, False)
            accum[t_idx, :] = out_vec

        summed = accum.sum(axis=0)
        start = col_idx * Num_Columns
        full_out[start:start + Num_Columns] = summed

    return n, i, j, full_out[:Cout]


def conv2d_tiles(x, w, Num_rows, Num_Columns, padding=0, mode="gs", checkboard=False,workers=8):
    N, Cin, H, W     = x.shape
    Cout, _, Kh, Kw  = w.shape
    # print("Input shape:", x.shape)
    # print("weight shape:", w.shape)
    # print(H, W, Kh, Kw)
    # print(Cout, Cin, N)
    # print(padding)
    Hout = H + 2*padding - Kh + 1
    Wout = W + 2*padding - Kw + 1

    # Zero-pad input
    x_p = torch.zeros((N, Cin, H + 2*padding, W + 2*padding), dtype=x.dtype, device=x.device)
    x_p[:, :, padding:padding+H, padding:padding+W] = x

    # Build crossbar_weights
    kernel_size = Kh * Kw
    cin_per_cross = Num_rows // kernel_size
    num_tiles_rows = int(np.ceil((kernel_size * Cin) / (cin_per_cross * kernel_size)))
    num_tiles_columns = int(np.ceil(Cout / Num_Columns))

    crossbar_weights = np.zeros((num_tiles_rows, num_tiles_columns, Num_rows, Num_Columns), dtype=float)
    for co in range(Cout):
        tile_j = co // Num_Columns
        col_idx = co % Num_Columns
        for ci in range(Cin):
            tile_i = ci // cin_per_cross
            id_mod = ci % cin_per_cross
            start = id_mod * kernel_size
            end   = start + kernel_size
            # flat_w = w[co, ci].view(-1).detach()
            flat_w = w[co, ci].view(-1).detach().numpy()
            crossbar_weights[tile_i, tile_j, start:end, col_idx] = flat_w

    # Build input_vec
    input_vec = np.zeros((N, Hout, Wout, num_tiles_rows, Num_rows), dtype=float)
    for n in range(N):
        for ci in range(Cin):
            tile_i = ci // cin_per_cross
            id_mod = ci % cin_per_cross
            start = id_mod * kernel_size
            end   = start + kernel_size
            for i in range(Hout):
                for j in range(Wout):
                    # patch = x_p[n, ci, i:i+Kh, j:j+Kw].contiguous().view(-1).detach()
                    patch = x_p[n, ci, i:i+Kh, j:j+Kw].contiguous().view(-1).detach().numpy()
                    input_vec[n, i, j, tile_i, start:end] = patch

    # Allocate output
    output = np.zeros((N, Cout, Hout, Wout), dtype=float)

    # Parallelize over (i,j) for each batch n using futures and as_completed
    for n in range(N):
        args_list = [
            (
                n, i, j,
                input_vec,
                crossbar_weights,
                Cout,
                num_tiles_columns,
                Num_rows,
                Num_Columns,
                mode,
                checkboard
            )
            for i in range(Hout) for j in range(Wout)
        ]
        disable_loggin = True
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_one_pixel, args) for args in args_list]
            for future in tqdm(as_completed(futures), total=len(futures),disable=disable_loggin):
                n_ret, i_ret, j_ret, vec_out = future.result()
                output[n_ret, :, i_ret, j_ret] = vec_out
    output = torch.from_numpy(output)
    output = output.float()  
    # output.dtype = torch.float32
    return output

def get_conv_output(pos_ref, neg_ref,Kh, Kw, CIN):
    I = pos_ref + neg_ref
    out = 2*I - Kh*Kw*CIN
    return out

def conv_inferenece(x,w, Num_rows, Num_Columns, padding=0, mode="gs", checkboard=False, workers=8):
    N, Cin, H, W     = x.shape
    Cout, _, Kh, Kw  = w.shape
    pos_inputs, neg_inputs = compliment(x)
    pos_filters, neg_filters = compliment(w)
    pos_cim = conv2d_tiles(pos_inputs,pos_filters,Num_rows,Num_Columns,padding=padding,mode=mode,checkboard=checkboard,workers=workers)
    neg_cim = conv2d_tiles(neg_inputs,neg_filters,Num_rows,Num_Columns,padding=padding,mode=mode,checkboard=checkboard,workers=workers)
    output = get_conv_output(pos_cim, neg_cim, Kh, Kw, Cin)
    return output

###########################################################################################

def _process_linear_tile(args):
    """
    Worker for one tile of the linear layer.
    Returns (b, j, out_vec) same shape as one_tile would produce.
    """
    b, i, j, N, num_tiles_columns, Num_rows, Num_Columns, vec, W, mode, checkboard = args
    if checkboard and j == (num_tiles_columns - 1):
        # apply checkerboard correction on the last column tile
        checkerboard_last_cols(W, Num_Columns - N % Num_Columns)
    _, out_vec = _task(((i, j), vec), W, Num_rows, Num_Columns, mode, False)
    return b, j, out_vec

def linear_parallel(x, w,
                    Num_rows, Num_Columns,
                    mode="gs", checkboard=False,
                    workers=8):
    """
    x: torch tensor or numpy array of shape (B, M)
    w: numpy array of shape (M, N)
    returns torch tensor of shape (B, N)
    """
    # ensure numpy
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    B, M = x_np.shape
    M2, N = w.shape
    assert M == M2, "Input/features mismatch"

    # how many tiles we need
    num_tiles_columns = int(np.ceil(N / Num_Columns))
    num_tiles_rows    = int(np.ceil(M / Num_rows))

    # build tiled weight array
    crossbar_weights = np.zeros((num_tiles_rows, num_tiles_columns, Num_rows, Num_Columns),
                                dtype=bool)
    for i in range(num_tiles_rows):
        for j in range(num_tiles_columns):
            r0, r1 = i*Num_rows, min((i+1)*Num_rows, M)
            c0, c1 = j*Num_Columns, min((j+1)*Num_Columns, N)
            w_tile = w[r0:r1, c0:c1]
            crossbar_weights[i, j, :w_tile.shape[0], :w_tile.shape[1]] = w_tile

    # build tiled input array
    input_vec = np.zeros((B, num_tiles_rows, Num_rows), dtype=bool)
    for b in range(B):
        for i in range(num_tiles_rows):
            r0, r1 = i*Num_rows, min((i+1)*Num_rows, M)
            input_vec[b, i, :r1-r0] = x_np[b, r0:r1]

    # prepare output accumulator (numpy)
    out = np.zeros((B, num_tiles_columns * Num_Columns), dtype=int)

    # pack all tile arguments
    tasks = []
    for b in range(B):
        for i in range(num_tiles_rows):
            for j in range(num_tiles_columns):
                vec = input_vec[b, i, :]
                W   = crossbar_weights[i, j, :, :]
                tasks.append((b, i, j, N, num_tiles_columns,
                              Num_rows, Num_Columns,
                              vec, W, mode, checkboard))

    # parallel execution
    with ProcessPoolExecutor(max_workers=workers) as exe:
        future_to_tile = {
            exe.submit(_process_linear_tile, args): args[:3]
            for args in tasks
        }
        for fut in as_completed(future_to_tile):
            b, j, out_vec = fut.result()
            start = j * Num_Columns
            end   = start + Num_Columns
            out[b, start:end] += out_vec

    # slice off any padding columns and wrap in torch.Tensor
    out = out[:, :N]
    out = torch.from_numpy(out)
    out = out.float()  
    # out.dtype = torch.float32

    return out

def get_fc_output(pos, neg, M):
    I = pos + neg
    out = 2*I - M
    return out

def linear_inferenece(x,w, Num_rows, Num_Columns, mode="gs", checkboard=False, workers=8):
    shape = w.shape
    pos_inputs, neg_inputs = compliment(x)
    pos_filters, neg_filters = compliment(w)
    pos_cim = linear_parallel(pos_inputs.detach().numpy(),pos_filters.detach().numpy().T, Num_rows, Num_Columns, mode=mode, checkboard=checkboard, workers=workers)
    neg_cim = linear_parallel(neg_inputs.detach().numpy(),neg_filters.detach().numpy().T, Num_rows, Num_Columns, mode=mode, checkboard=checkboard, workers=workers)
    output = get_fc_output(pos_cim, neg_cim, shape[1])
    return output

################################################################################################################
