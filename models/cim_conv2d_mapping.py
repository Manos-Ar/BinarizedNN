import sys
sys.path.append('/shares/bulk/earapidis/dev/Fast-Crossbar-Sim/python')
from crossbar import _task
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import time

def checkerboard_last_cols(arr: torch.Tensor, C: int) -> torch.Tensor:
    n, m = arr.shape
    if C == 0:
        return arr.clone()  # return a copy even if unchanged
    if C > m:
        raise ValueError("C cannot be larger than the number of columns m.")
    
    arr_copy = arr.clone()
    rows = torch.arange(n).view(-1, 1)                
    cols = torch.arange(m - C, m).view(1, -1)         

    pattern = (rows + cols) % 2
    arr_copy[:, -C:] = pattern
    return arr_copy

def compliment(x):
    x = x.clone()
    neg = -1*x
    pos = x

    pos[pos==-1] = 0
    neg[neg==-1] = 0
    return pos, neg

def get_conv2d_output(I,Kh, Kw, CIN):
    out = 2*I - Kh*Kw*CIN
    return out

def map_conv2d(x,w,padding=0):
    _COUT_, _CIN_, _Kh_, _Kw_ = w.shape
    _N_, _CIN_, _Hi_, _Wi_ = x.shape
    pos_w, neg_w = compliment(w)
    mapped_w = torch.empty((_COUT_,_CIN_,2*(_Kh_*_Kw_)))

    pos_w = pos_w.reshape(_COUT_,_CIN_,-1)
    neg_w = neg_w.reshape(_COUT_,_CIN_,-1)
    kernel_size = _Kh_*_Kw_
    for i in range(kernel_size):

        mapped_w[:,:,2*i] = pos_w[:,:,i]
        mapped_w[:,:,2*i+1] = neg_w[:,:,i]
    
    _Hout_ = _Hi_ + 2*padding - _Kh_ + 1
    _Wout_ = _Wi_ + 2*padding - _Kw_ + 1
    
    mapped_x = torch.empty((_N_,_Hout_,_Wout_,_CIN_,2*kernel_size))
    pos_x, neg_x = compliment(x)
    
    for ii in range(_Hout_):
        for jj in range(_Wout_):
            _pos_ = pos_x[:,:,ii:ii+_Kh_,jj:jj+_Kw_]
            _neg_ = neg_x[:,:,ii:ii+_Kh_,jj:jj+_Kw_]
            # print(_pos_.shape)
            _pos_ = _pos_.reshape(_N_,_CIN_,-1)
            _neg_ = _neg_.reshape(_N_,_CIN_,-1)
            one_kernel = torch.empty(_N_,_CIN_,2*kernel_size)
            for z in range(kernel_size):
                one_kernel[:,:,2*z]=_pos_[:,:,z]
                one_kernel[:,:,2*z+1]=_neg_[:,:,z]
            mapped_x[:,ii,jj,:,:] = one_kernel
    return mapped_x,mapped_w


def run_tile(ii,jj,cx, tmp_x_np, tmp_w_np, Num_rows, Num_Columns, columns_per_crossbar, Total_dim, mode,transient):
    tmp_w_np = checkerboard_last_cols(tmp_w_np, tmp_w_np.shape[1] - columns_per_crossbar)
    _, _out_np = _task(((ii,jj), tmp_x_np), tmp_w_np, Num_rows, Num_Columns, mode, transient)

    column_start_idx = cx * columns_per_crossbar
    column_end_idx = (cx + 1) * columns_per_crossbar
    # column_end_idx = Total_dim if tmp_w_np.shape[1] - columns_per_crossbar < columns_per_crossbar else (cx + 1) * columns_per_crossbar
    return ((ii, jj, column_start_idx, column_end_idx), _out_np[:, :columns_per_crossbar])

def cim_conv_2d(crossbar_inputs, crossbar_weights, _COUT_, mode, Total_dim, transient,max_workers=None):
    _N_, _HOUT_, _WOUT_, crossbar_y, Num_rows = crossbar_inputs.shape
    _, crossbar_x, _, Num_Columns = crossbar_weights.shape
    output_conv_2d = torch.zeros(_N_, _COUT_, _HOUT_, _WOUT_)

    columns_per_crossbar = math.floor(_COUT_ / crossbar_x)

    # Preconvert weights to NumPy once (this is reused for each (ii, jj))
    # weights_np = [[crossbar_weights[cy, cx].numpy() for cx in range(crossbar_x)] for cy in range(crossbar_y)]

    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ii in range(_HOUT_):
            for jj in range(_WOUT_):

                for cy in range(crossbar_y):
                    for cx in range(crossbar_x):
                        tmp_x = crossbar_inputs[:, ii, jj, cy, :].numpy()
                        tmp_w = crossbar_weights[cy][cx]
                        # tmp_w = weights_np[cy][cx]
                        tasks.append((ii,jj, cx, tmp_x, tmp_w, Num_rows, Num_Columns, columns_per_crossbar, Total_dim, mode, transient))

        futures = [executor.submit(run_tile, *t) for t in tasks]
        # for f in tqdm(as_completed(futures),total=len(tasks)):
        for f in tqdm(as_completed(futures),total=len(tasks),disable=True):
            (ii,jj, col_start, col_end), out_np = f.result()
            out = torch.from_numpy(out_np)
            output_conv_2d[:, col_start:col_end, ii, jj] += out

    return output_conv_2d

def _conv2d_tile_(x,w,Num_rows,Num_Columns,mode,max_workers,transient=False):
    _N_,_HOUT_, _WOUT_,_CIN_, _kernel_size_ = x.shape
    _COUT_, _CIN_, _kernel_size_ = w.shape
    output_conv_2d = torch.zeros(_N_,_COUT_,_HOUT_,_WOUT_)

    whole_input_size = _CIN_*_kernel_size_
    # print(f"total inputs: {whole_input_size}")
    crossbar_y = math.ceil((_CIN_*_kernel_size_)/Num_rows)
    crossbar_x = math.ceil(_COUT_/Num_Columns)

    crossbar_weights = torch.zeros((crossbar_y,crossbar_x,Num_rows,Num_Columns))
    crossbar_inputs = torch.zeros((_N_,_HOUT_,_WOUT_,crossbar_y,Num_rows))

    rows_per_crossbar = math.floor(whole_input_size/crossbar_y)

    flatten_x = x.reshape(*x.shape[:-2],-1)
    # print(flatten_x.shape)
    flatten_w = w.reshape(*w.shape[:-2],-1).T
    # print(flatten_w.shape)
    # print(crossbar_inputs.shape)
    # print(f"weights shape: {crossbar_weights.shape}")
    # print(f"inputs shape: {crossbar_inputs.shape}")

    columns_per_crossbar = math.floor(_COUT_/crossbar_x)

    for ii in range(crossbar_y):
        row_start_idx = ii*rows_per_crossbar
        # if ii==crossbar_y-1:
        #     row_end_idx = flatten_x.shape[-1]
        # else:
        row_end_idx = (ii+1)*rows_per_crossbar
        # print(start_idx,end_idx)
        crossbar_inputs[:,:,:,ii,:rows_per_crossbar] = flatten_x[:,:,:,row_start_idx:row_end_idx]
    # for ii in range(0,whole_input_size,step=inpu)

        for jj in range(crossbar_x):
            column_start_idx = jj*columns_per_crossbar
            # if jj==crossbar_x-1:
            #     column_end_idx=flatten_w.shape[-1]
            # else:
            column_end_idx = (jj+1)*columns_per_crossbar
            
            crossbar_weights[ii,jj,:rows_per_crossbar,:columns_per_crossbar] = flatten_w[row_start_idx:row_end_idx,column_start_idx:column_end_idx]


    output_conv_2d = cim_conv_2d(crossbar_inputs,crossbar_weights,_COUT_,mode,flatten_w.shape[-1],transient=transient,max_workers=max_workers)
    return output_conv_2d


def conv2d_tile(x,w,Num_rows,Num_Columns,mode,max_workers,transient=False):
    _N_, _CIN_, _Hi_, _Wi_ = x.shape
    _COUT_, _CIN_, _Kh_, _Kw_ = w.shape
    mapped_x, mapped_w = map_conv2d(x,w)
    output_conv_2d = _conv2d_tile_(mapped_x,mapped_w,Num_rows,Num_Columns,mode,max_workers,transient)
    output_conv_2d = get_conv2d_output(output_conv_2d,_Kh_,_Kw_,_CIN_)
    
    return output_conv_2d


def con2d(x,w,Num_rows,Num_Columns,mode,max_workers,transient=False):
    padding = 0
    _N_, _CIN_, _Hi_, _Wi_ = x.shape
    _COUT_, _CIN_, _Kh_, _Kw_ = w.shape
    _HOUT_ = _Hi_ + 2*padding - _Kh_ + 1
    _WOUT_ = _Wi_ + 2*padding - _Kw_ + 1
    output_conv_2d = torch.zeros(_N_,_COUT_,_HOUT_,_WOUT_)
    for idx in range(_N_):
        tmp_x = x[idx].unsqueeze(0)
        output_conv_2d[idx,:,:,:] = conv2d_tile(tmp_x,w,Num_rows,Num_Columns,mode,max_workers,transient=False)
    return output_conv_2d

