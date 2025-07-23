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

def map_conv2d_regular_mapping_one(x,w,padding=0):
    _N_, _CIN_, _Hi_, _Wi_ = x.shape
    _COUT_, _CIN_, _Kh_, _Kw_ = w.shape

    mapped_w = w.reshape(_COUT_,_CIN_,-1)
    kernel_size = _Kh_*_Kw_   
    _Hout_ = _Hi_ + 2*padding - _Kh_ + 1
    _Wout_ = _Wi_ + 2*padding - _Kw_ + 1
    
    mapped_x = torch.empty((_N_,_Hout_,_Wout_,_CIN_,kernel_size))
    
    for ii in range(_Hout_):
        for jj in range(_Wout_):
            _x_ = x[:,:,ii:ii+_Kh_,jj:jj+_Kw_]
            mapped_x[:,ii,jj,:,:] = _x_.reshape(_N_,_CIN_,-1)

    return mapped_x,mapped_w

def map_conv2d_regular_mapping(x,w,padding=0):
    pos_x, neg_x = compliment(x)
    pos_w, neg_w = compliment(w)
    pos_x, pos_w = map_conv2d_regular_mapping_one(pos_x,pos_w,padding)
    neg_x, neg_w = map_conv2d_regular_mapping_one(neg_x,neg_w,padding)

    return ((pos_x,pos_w),(neg_x,neg_w))

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


def parallel_conv_kernels(crossbar_inputs, crossbar_weights, _COUT_, mode,transient,max_workers=None):
    _N_, _HOUT_, _WOUT_, crossbar_y, Num_rows = crossbar_inputs.shape
    _, crossbar_x, _, Num_Columns = crossbar_weights.shape
    output_conv_2d = torch.zeros(_N_, _COUT_, _HOUT_, _WOUT_)

    columns_per_crossbar = math.ceil(_COUT_/crossbar_x)

    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ii in range(_HOUT_):
            for jj in range(_WOUT_):
                for cy in range(crossbar_y):
                    for cx in range(crossbar_x):
                        x = crossbar_inputs[:, ii, jj, cy, :].numpy()
                        w = crossbar_weights[cy][cx]
                        _vec_ = ((ii,jj,cx),x)
                        args = [_vec_, w, Num_rows, Num_Columns, mode, transient]
                        tasks.append(args)

        futures = [executor.submit(_task, *t) for t in tasks]
        for f in tqdm(as_completed(futures),total=len(tasks),disable=True):
            (ii,jj, cx), out_np = f.result()
            out = torch.from_numpy(out_np)
            column_start_idx = cx*columns_per_crossbar
            if cx==crossbar_x-1:
                column_end_idx=_COUT_
            else:
                column_end_idx = (cx+1)*columns_per_crossbar
            output_conv_2d[:, column_start_idx:column_end_idx, ii, jj] += out[:,:column_end_idx-column_start_idx]

    return output_conv_2d


def get_inputs_to_cim(x,Num_rows):
    _N_,_HOUT_, _WOUT_,_CIN_, _kernel_size_ = x.shape

    whole_input_size = _CIN_*_kernel_size_
    # print(f"total inputs: {whole_input_size}")
    crossbar_y = math.ceil((_CIN_*_kernel_size_)/Num_rows)

    crossbar_inputs = torch.zeros((_N_,_HOUT_,_WOUT_,crossbar_y,Num_rows))

    rows_per_crossbar = math.ceil(whole_input_size/crossbar_y)

    flatten_x = x.reshape(_N_,_HOUT_,_WOUT_,whole_input_size)

    for cy in range(crossbar_y):
        row_start_idx = cy*rows_per_crossbar
        if cy==crossbar_y-1:
            row_end_idx = whole_input_size
        else:
            row_end_idx = (cy+1)*rows_per_crossbar
        crossbar_inputs[:,:,:,cy,:row_end_idx-row_start_idx] = flatten_x[:,:,:,row_start_idx:row_end_idx]
    return crossbar_inputs

def get_weights_to_cim(w,Num_rows,Num_columns,checkboard=False):
    _COUT_, _CIN_, _kernel_size_ = w.shape

    whole_input_size = _CIN_*_kernel_size_
    # print(f"total inputs: {whole_input_size}")
    crossbar_y = math.ceil((_CIN_*_kernel_size_)/Num_rows)
    crossbar_x = math.ceil(_COUT_/Num_columns)

    crossbar_weights = torch.zeros((crossbar_y,crossbar_x,Num_rows,Num_columns))
    rows_per_crossbar = math.ceil(whole_input_size/crossbar_y)
    flatten_w = w.reshape(_COUT_,whole_input_size).T

    columns_per_crossbar = math.ceil(_COUT_/crossbar_x)

    for cy in range(crossbar_y):
        row_start_idx = cy*rows_per_crossbar
        if cy==crossbar_y-1:
            row_end_idx = whole_input_size
        else:
            row_end_idx = (cy+1)*rows_per_crossbar

        for cx in range(crossbar_x):
            column_start_idx = cx*columns_per_crossbar
            if cx==crossbar_x-1:
                column_end_idx=_COUT_
            else:
                column_end_idx = (cx+1)*columns_per_crossbar
            
            crossbar_weights[cy,cx,:row_end_idx-row_start_idx,:column_end_idx-column_start_idx] = flatten_w[row_start_idx:row_end_idx,column_start_idx:column_end_idx]
            if checkboard:
                crossbar_weights[cy,cx] = checkerboard_last_cols(crossbar_weights[cy,cx], Num_columns - columns_per_crossbar)
    return crossbar_weights


def conv2d_to_cim(x,w,Num_rows,Num_columns,mode,max_workers,checkboard,transient):
    _COUT_, _CIN_, _kernel_size_ = w.shape

    crossbar_inputs = get_inputs_to_cim(x,Num_rows)
    crossbar_weights = get_weights_to_cim(w,Num_rows,Num_columns,checkboard)
    output_conv_2d = parallel_conv_kernels(crossbar_inputs,crossbar_weights,_COUT_,mode,transient=transient,max_workers=max_workers)
    return output_conv_2d

def conv2d_one_input(x,w,Num_rows,Num_Columns,mode,max_workers,transient,checkboard,mapping):
    _N_, _CIN_, _Hi_, _Wi_ = x.shape
    _COUT_, _CIN_, _Kh_, _Kw_ = w.shape
    if mapping:
        mapped_x, mapped_w = map_conv2d(x,w)
        output_conv_2d = conv2d_to_cim(mapped_x,mapped_w,Num_rows,Num_Columns,mode=mode,max_workers=max_workers,checkboard=checkboard,transient=transient)
        output_conv_2d = get_conv2d_output(output_conv_2d,_Kh_,_Kw_,_CIN_)
    else:    
        (pos_x,pos_w), (neg_x,neg_w) = map_conv2d_regular_mapping(x,w,padding=0)
        pos_output_conv_2d = conv2d_to_cim(pos_x,pos_w,Num_rows,Num_Columns,mode=mode,max_workers=max_workers,checkboard=checkboard,transient=transient)
        neg_output_conv_2d = conv2d_to_cim(neg_x,neg_w,Num_rows,Num_Columns,mode=mode,max_workers=max_workers,checkboard=checkboard,transient=transient)
        output_conv_2d = pos_output_conv_2d+neg_output_conv_2d
        output_conv_2d = get_conv2d_output(output_conv_2d,_Kh_,_Kw_,_CIN_)
    
    return output_conv_2d


def conv2d(x,w,Num_rows,Num_Columns,mode,max_workers,transient,checkboard,mapping):
    padding = 0
    _N_, _CIN_, _Hi_, _Wi_ = x.shape
    _COUT_, _CIN_, _Kh_, _Kw_ = w.shape
    _HOUT_ = _Hi_ + 2*padding - _Kh_ + 1
    _WOUT_ = _Wi_ + 2*padding - _Kw_ + 1
    output_conv_2d = torch.zeros(_N_,_COUT_,_HOUT_,_WOUT_)
    for idx in range(_N_):
        tmp_x = x[idx].unsqueeze(0)
        output_conv_2d[idx,:,:,:] = conv2d_one_input(tmp_x,w,Num_rows,Num_Columns,mode=mode,max_workers=max_workers,transient=transient,checkboard=checkboard,mapping=mapping)
    return output_conv_2d

