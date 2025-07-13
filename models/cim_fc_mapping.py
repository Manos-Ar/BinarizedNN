import sys
sys.path.append('/shares/bulk/earapidis/dev/Fast-Crossbar-Sim/python')
from crossbar import _task
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

from models.cim_conv2d_mapping import compliment, checkerboard_last_cols
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

def  fc_linear(crossbar_inputs,crossbar_weights,N,mode,max_workers,transient):
    crossbar_y,crossbar_x,Num_rows,Num_Columns = crossbar_weights.shape
    _N_, crossbar_y, Num_rows = crossbar_inputs.shape
    output = torch.zeros(_N_,N)
    columns_per_crossbar = math.floor(N/crossbar_x)
    
    tasks = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ii in range(crossbar_y):
            for jj in range(crossbar_x):
                column_start_idx = jj*columns_per_crossbar
                column_end_idx = (jj+1)*columns_per_crossbar
                tmp_x = crossbar_inputs[:,ii]
                tmp_w = crossbar_weights[ii][jj]
                checkerboard_last_cols(tmp_w,Num_Columns-columns_per_crossbar)
                # out_matmul = torch.matmul(tmp_x,tmp_w)
                args = (((ii,jj),tmp_x),tmp_w,Num_rows,Num_Columns,mode,transient)
                tasks.append(args)
        futures = [executor.submit(_task, *t) for t in tasks]
        for f in tqdm(as_completed(futures),total=len(tasks)):
            (ii,jj), out_matmul = f.result()
            column_start_idx = jj*columns_per_crossbar
            column_end_idx = (jj+1)*columns_per_crossbar
            out_matmul = torch.from_numpy(out_matmul)
            output[:,column_start_idx:column_end_idx] += out_matmul[:,:columns_per_crossbar]

    return output

def _fc_tile_(x,w, Num_rows,Num_Columns,mode,max_workers,transient):
    _N_ , M = x.shape
    M , N = w.shape

    crossbar_y = math.ceil(M/Num_rows)
    crossbar_x = math.ceil(N/Num_Columns)
    # print("crossbar grid",crossbar_y,crossbar_x)

    crossbar_inputs = torch.zeros((_N_,crossbar_y,Num_rows))
    crossbar_weights = torch.zeros((crossbar_y,crossbar_x,Num_rows,Num_Columns))

    rows_per_crossbar = math.floor(M/crossbar_y)
    columns_per_crossbar = math.floor(N/crossbar_x)

    # print(rows_per_crossbar, columns_per_crossbar)

    for ii in range(crossbar_y):
        row_start_idx = ii*rows_per_crossbar
        # if ii==crossbar_y-1:
        #     row_end_idx = M
        # else:
        row_end_idx = (ii+1)*rows_per_crossbar
        crossbar_inputs[:,ii,:rows_per_crossbar] = x[:,row_start_idx:row_end_idx]
    # for ii in range(0,whole_input_size,step=inpu)
        # print(row_start_idx,row_end_idx)

        for jj in range(crossbar_x):
            column_start_idx = jj*columns_per_crossbar
            # if jj==crossbar_x-1:
            #     column_end_idx=N
            # else:
            column_end_idx = (jj+1)*columns_per_crossbar
            # print(column_start_idx,column_end_idx)
            
            crossbar_weights[ii,jj,:rows_per_crossbar,:columns_per_crossbar] = w[row_start_idx:row_end_idx,column_start_idx:column_end_idx]
        
    # print(f"crossbar weigths : {crossbar_weights.shape}")
    # print(f"crossbar inputs : {crossbar_inputs.shape}")
    return fc_linear(crossbar_inputs,crossbar_weights,N,mode,max_workers,transient)

def fc_tile(x,w, Num_rows,Num_Columns,mode,max_workers,transient):
    # w = w.T
    M , N = w.shape
    _N_ , M = x.shape
    mapped_x, mapped_w = map_fc(x,w)
    output_fc = _fc_tile_(mapped_x,mapped_w,Num_rows,Num_Columns,mode,max_workers,transient)
    output_fc = get_fc_output(output_fc,M)
    return output_fc

def fc(x,w, Num_rows,Num_Columns,mode,max_workers,transient):
    # w = w.T
    M , N = w.shape
    _N_ , M = x.shape
    output_fc = torch.empty((_N_,N))
    for idx in range(_N_):
        tmp_x = x[idx].unsqueeze(0)
        output_fc[idx,:] = fc_tile(tmp_x,w, Num_rows,Num_Columns,mode,max_workers,transient)
    return output_fc