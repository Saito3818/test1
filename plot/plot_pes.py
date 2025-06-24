#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import re
import cv_hist_plot as chp
from tqdm import tqdm

def get_logfile(logfile):
    with open(logfile) as o:
        body = o.read()
    data_key = re.search('Step.+?\n', body).group()
    data_list = re.findall(f'{data_key}([\s\d\.\-e\+]+)', body, re.DOTALL)
    data_key = data_key.strip().split()
    data_dict = {key: np.empty(0, dtype=float) for key in ['Step', 'PotEng']}
    time_init = 0
    for data in data_list:
        data = np.array(data.strip().split(),
                        dtype=float).reshape(-1, len(data_key))
        for key, value in data_dict.items():
            key_idx = data_key.index(key)
            if key == 'Step':
                if np.any(data[:, key_idx] == 0) & (value.size != 0):
                    time_init = value[-1]
                data[:, key_idx] += time_init
            data_dict[key] = np.concatenate((value, data[:, key_idx]))
    return data_dict

def get_colvar(colvar, cv1, cv2):
    with open(colvar) as o:
        body = o.read()
    data_key = re.search('#! FIELDS .+?\n', body).group()
    data_list = re.findall(f'{data_key}([\s\d\.\-e\+]+)', body, re.DOTALL)
    data_key = data_key.strip().split()[2:]
    data_dict = {key: np.empty(0, dtype=float) for key in ['time', cv1, cv2]}
    time_init = 0
    for data in data_list:
        data = np.array(data.strip().split(),
                        dtype=float).reshape(-1, len(data_key))
        for key, value in data_dict.items():
            key_idx = data_key.index(key)
            if key == 'time':
                if np.any(data[:, key_idx] == 0) & (value.size != 0):
                    time_init = value[-1]
                data[:, key_idx] += time_init
            data_dict[key] = np.concatenate((value, data[:, key_idx]))
    return data_dict

def save_debag(log_data, colvar_data, debag_file="log_colvar2.npy"):
    log_step = np.unique(log_data['Step'])
    data = np.empty((log_step.size, 4), dtype=float)  # step, pe, cv1, cv2
    data[:, 0] = log_step
    idx = 0
    for step in tqdm(log_step):
        data[idx, 1] = log_data['PotEng'][log_data['Step'] == step][-1]
        colvar_idx = np.where(colvar_data['time'] == step)[0]
        data[idx, 2] = colvar_data[args.cv1][colvar_idx][0]  # some times
        data[idx, 3] = colvar_data[args.cv2][colvar_idx][0]
        idx += 1
    np.save(debag_file, data)
    print(f'{debag_file} created!')
    return data

def make_meshdata(data, cv_bin):
    cv_array = cv_bin[:-1] + (cv_bin[1] - cv_bin[0])
    X = np.tile(cv_array, cv_array.size).reshape(cv_array.size, -1)
    Y = np.repeat(cv_array, cv_array.size).reshape(cv_array.size, -1)
    Z = np.zeros(X.shape, dtype=float)
    idx1 = 0
    for cv1 in tqdm(cv_bin[:-1]):
        for idx2, cv2 in enumerate(cv_bin[:-1]):
            cv1_idx = np.where((cv1 <= data[:, 2])
                               & (cv_bin[idx1+1] > data[:, 2]))[0]
            cv2_idx = np.where((cv2 <= data[:, 3])
                               & (cv_bin[idx2+1] > data[:, 3]))[0]
            cv_idx = cv1_idx[np.isin(cv1_idx, cv2_idx)]
            if cv_idx.size != 0:
                Z[idx2, idx1] = data[cv_idx, 1].min()
        idx1 += 1
    Z[Z == 0] = np.nan
    return X, Y, Z

if __name__ == "__main__":
    import argparse
    par = argparse.ArgumentParser(description="test")
    par.add_argument('-l', '--log', default="test_1000.log")
    par.add_argument('-c', '--colvar', default="test_1000.colvar")
    par.add_argument('-x', '--cv1', default="RD22")
    par.add_argument('-y', '--cv2', default="RD33")
    par.add_argument('-d', '--debag_file', default=None, help="log_colvar.npy")
    par.add_argument('-w', '--window', type=float, nargs=2, default=[-40, 180])
    par.add_argument('-b', '--num_bin', type=int, default=321)
    par.add_argument('-o', '--outpdf', default='PES.pdf')
    args = par.parse_args()


    if args.debag_file == None:
        
        # log_data = get_logfile(args.log)
        
        log_data = {}
        log_data['Step'] = np.load('step.npy')
        log_data['PotEng'] = np.load('pe.npy')
        # print(log_data['Step'])

        # colvar_data = get_colvar(args.colvar, args.cv1, args.cv2)
        # print(colvar_data['time'])
        # np.save('time.npy', colvar_data['time'])
        # np.save('cv1.npy', colvar_data[args.cv1])
        # np.save('cv2.npy', colvar_data[args.cv2])

        colvar_data = {}
        colvar_data['time'] = np.load('time.npy')
        colvar_data[args.cv1] = np.load('cv1.npy')
        colvar_data[args.cv2] = np.load('cv2.npy')

        data = save_debag(log_data, colvar_data)
    else:
        data = np.load(args.debag_file)  # time, energy, cv1, cv2
        print(f'{args.debag_file} read!')

    cv_bin = np.linspace(args.window[0], args.window[1], args.num_bin)
    X, Y, Z = make_meshdata(data, cv_bin)
    np.save('X2.npy', X)
    np.save('Y2.npy', Y)
    np.save('Z2.npy', Z)
    exit()

    chp.plot_2dim(X, Y, Z, 'CV1', 'CV2', title='', c_map='jet',
                  save_fig=args.outpdf, contourf=False)
    print(f"{args.outpdf} created")
