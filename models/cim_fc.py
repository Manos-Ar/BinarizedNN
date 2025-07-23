import sys
sys.path.append('/shares/bulk/earapidis/dev/Fast-Crossbar-Sim/python')
from crossbar import _task
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

from models.cim_conv import compliment, checkerboard_last_cols
import time
def get_fc_output(I, M):
    out = 2*I - M
    return out

def map_fc(x,w):
    M, N = w.shape
    _N_, M = x.shape
    pos_w, neg_w = compliment(w) 
    pos_x, neg_x = compliment(x)

    mapped_w = torch.empty((2*M,N))
    mapped_x = torch.empty((_N_,2*M))

    for idx in range(M):
        _pos_w_ = pos_w[idx]
        _neg_w_ = neg_w[idx]
        mapped_w[2*idx]=_pos_w_
        mapped_w[2*idx+1]=_neg_w_

        _pos_x_ = pos_x[:,idx]
        _neg_x_ = neg_x[:,idx]

        mapped_x[:,2*idx] = _pos_x_
        mapped_x[:,2*idx+1] = _neg_x_

    return mapped_x,mapped_w

def parallel_fc_kernels(crossbar_inputs,crossbar_weights,N,mode,max_workers,transient):
    crossbar_y,crossbar_x,Num_rows,Num_columns = crossbar_weights.shape
    _N_, crossbar_y, Num_rows = crossbar_inputs.shape
    output_fc = torch.zeros(_N_,N)
    columns_per_crossbar = math.ceil(N/crossbar_x)
    
    tasks = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for cy in range(crossbar_y):
            for cx in range(crossbar_x):
                x = crossbar_inputs[:,cy]
                w = crossbar_weights[cy][cx]
                _vec_ = (cx,x)
                args = [_vec_,w,Num_rows,Num_columns,mode,transient]
                tasks.append(args)
        futures = [executor.submit(_task, *t) for t in tasks]
        for f in tqdm(as_completed(futures),total=len(tasks),disable=True):
            cx, output = f.result()
            column_start_idx = cx*columns_per_crossbar
            if cx==crossbar_x-1:
                column_end_idx=N
            else:
                column_end_idx = (cx+1)*columns_per_crossbar
            output = torch.from_numpy(output)
            output_fc[:,column_start_idx:column_end_idx] += output[:,:column_end_idx-column_start_idx]

    return output_fc



def get_inputs_to_cim(x,Num_rows):
    _N_ , M = x.shape
    crossbar_y = math.ceil(M/Num_rows)
    crossbar_inputs = torch.zeros((_N_,crossbar_y,Num_rows))
    rows_per_crossbar = math.ceil(M/crossbar_y)

    for cy in range(crossbar_y):
        row_start_idx = cy*rows_per_crossbar
        if cy==crossbar_y-1:
            row_end_idx = M
        else:
            row_end_idx = (cy+1)*rows_per_crossbar
        crossbar_inputs[:,cy,:row_end_idx-row_start_idx] = x[:,row_start_idx:row_end_idx]    
    return crossbar_inputs

def get_weights_to_cim(w,Num_rows,Num_columns,checkboard):
    M , N = w.shape
    crossbar_y = math.ceil(M/Num_rows)
    crossbar_x = math.ceil(N/Num_columns)
    # print("crossbar grid",crossbar_y,crossbar_x)

    crossbar_weights = torch.zeros((crossbar_y,crossbar_x,Num_rows,Num_columns))

    rows_per_crossbar = math.ceil(M/crossbar_y)
    columns_per_crossbar = math.ceil(N/crossbar_x)

    # print(rows_per_crossbar, columns_per_crossbar)

    for cy in range(crossbar_y):
        row_start_idx = cy*rows_per_crossbar
        if cy==crossbar_y-1:
            row_end_idx = M
        else:
            row_end_idx = (cy+1)*rows_per_crossbar

        for cx in range(crossbar_x):
            column_start_idx = cx*columns_per_crossbar
            if cx==crossbar_x-1:
                column_end_idx=N
            else:
                column_end_idx = (cx+1)*columns_per_crossbar
            crossbar_weights[cy,cx,:row_end_idx-row_start_idx,:column_end_idx-column_start_idx] = w[row_start_idx:row_end_idx,column_start_idx:column_end_idx]    
            if checkboard:
                crossbar_weights[cy,cx] = checkerboard_last_cols(crossbar_weights[cy,cx], Num_columns - columns_per_crossbar)
    return crossbar_weights

def fc_to_cim(x,w, Num_rows,Num_columns,mode,max_workers,transient,checkboard):
    M , N = w.shape

    crossbar_inputs = get_inputs_to_cim(x,Num_rows)        
    crossbar_weights = get_weights_to_cim(w,Num_rows,Num_columns,checkboard)
    # print(f"crossbar weigths : {crossbar_weights.shape}")
    # print(f"crossbar inputs : {crossbar_inputs.shape}")
    return parallel_fc_kernels(crossbar_inputs,crossbar_weights,N,mode=mode,max_workers=max_workers,transient=transient)

def fc_one_input(x,w, Num_rows,Num_columns,mode,max_workers,transient,checkboard,mapping):
    M , N = w.shape
    _N_ , M = x.shape
    if mapping:

        mapped_x, mapped_w = map_fc(x,w)
        output_fc = fc_to_cim(mapped_x,mapped_w,Num_rows,Num_columns,mode,max_workers,transient,checkboard)
    else:
        pos_x, neg_x = compliment(x)
        pos_w, neg_w = compliment(w)
        pos_output_fc = fc_to_cim(pos_x,pos_w,Num_rows,Num_columns,mode=mode,max_workers=max_workers,transient=transient,checkboard=checkboard)
        neg_output_fc = fc_to_cim(neg_x,neg_w,Num_rows,Num_columns,mode=mode,max_workers=max_workers,transient=transient,checkboard=checkboard)
        output_fc = pos_output_fc + neg_output_fc

    output_fc = get_fc_output(output_fc,M)
    return output_fc

def fc(x,w, Num_rows,Num_columns,mode,max_workers,transient,checkboard,mapping):
    M , N = w.shape
    _N_ , M = x.shape
    output_fc = torch.empty((_N_,N))
    for idx in range(_N_):
        tmp_x = x[idx].unsqueeze(0)
        output_fc[idx,:] = fc_one_input(tmp_x,w, Num_rows,Num_columns,mode=mode,max_workers=max_workers,transient=transient,checkboard=checkboard,mapping=mapping)
    return output_fc