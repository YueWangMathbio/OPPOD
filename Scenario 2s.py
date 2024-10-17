#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:03:37 2024

@author: yuewang
"""

import numpy as np
import functools
from itertools import product
from multiprocessing import Pool
import multiprocessing
from scipy.io import savemat
from DP_Alg import TS, MP, CVP, MCS

a_min = 8.0
a_max = 10.0
b_min = 4.0
b_max = 5.0
p_min = 0.7
p_max = 1.25
a_true = 9.0
b_true = 4.5
noise_true = 0.1

def TS_wrap_s(unused_index, noise_off, weight_off, t_off, t_on, b_off, a_off):
    del unused_index
    return TS(t_off, t_on, a_off, b_off, noise_off, weight_off)

def TS_rep_multiprocess_s(total_run, t_off, t_on, s_linspace, b_off, a_off, weight_linspace):
    full_combination = list(product(np.arange(total_run), s_linspace, weight_linspace))
    with Pool(multiprocessing.cpu_count() - 1) as p:
        output = p.starmap(functools.partial(TS_wrap_s, t_off=t_off, t_on=t_on, b_off=b_off, a_off=a_off), full_combination)
    output_reshape = np.reshape(output, (total_run, len(s_linspace), len(weight_linspace)))
    output_final = np.average(output_reshape, axis=0)
    return output_final

def MP_wrap_s(unused_index, noise_off, weight_off, t_off, t_on, b_off, a_off):
    del unused_index
    return MP(t_off, t_on, a_off, b_off, noise_off, weight_off)

def MP_rep_multiprocess_s(total_run, t_off, t_on, s_linspace, b_off, a_off, weight_linspace):
    full_combination = list(product(np.arange(total_run), s_linspace, weight_linspace))
    with Pool(multiprocessing.cpu_count() - 1) as p:
        output = p.starmap(functools.partial(MP_wrap_s, t_off=t_off, t_on=t_on, b_off=b_off, a_off=a_off), full_combination)
    output_reshape = np.reshape(output, (total_run, len(s_linspace), len(weight_linspace)))
    output_final = np.average(output_reshape, axis=0)
    return output_final

def CVP_wrap_s(unused_index, noise_off, weight_off, t_off, t_on, b_off, a_off):
    del unused_index
    return CVP(t_off, t_on, a_off, b_off, noise_off, weight_off)

def CVP_rep_multiprocess_s(total_run, t_off, t_on, s_linspace, b_off, a_off, weight_linspace):
    full_combination = list(product(np.arange(total_run), s_linspace, weight_linspace))
    with Pool(multiprocessing.cpu_count() - 1) as p:
        output = p.starmap(functools.partial(CVP_wrap_s, t_off=t_off, t_on=t_on, b_off=b_off, a_off=a_off), full_combination)
    output_reshape = np.reshape(output, (total_run, len(s_linspace), len(weight_linspace)))
    output_final = np.average(output_reshape, axis=0)
    return output_final

def MCS_wrap_s(unused_index, noise_off, weight_off, t_off, t_on, b_off, a_off):
    del unused_index
    return MCS(t_off, t_on, a_off, b_off, noise_off, weight_off)

def MCS_rep_multiprocess_s(total_run, t_off, t_on, s_linspace, b_off, a_off, weight_linspace):
    full_combination = list(product(np.arange(total_run), s_linspace, weight_linspace))
    with Pool(multiprocessing.cpu_count() - 1) as p:
        output = p.starmap(functools.partial(MCS_wrap_s, t_off=t_off, t_on=t_on, b_off=b_off, a_off=a_off), full_combination)
    output_reshape = np.reshape(output, (total_run, len(s_linspace), len(weight_linspace)))
    output_final = np.average(output_reshape, axis=0)
    return output_final

def main():
    t_off = 10
    t_on = 10
    total_run = 1 # repeat the simulation
    accuracy = 0.01
    weight_acc = 0.02
    scenario = 2
    if scenario == 1:
        weight_min = 1.0
        filehead = 's1s_'
        md = 's1s'
    if scenario == 2:
        weight_min = 0.0
        filehead = 's2s_'
        md = 's2s'
    weight_max = 1.0
    s_min = 0.01
    s_max = 1.0
    grid_num = int((s_max - s_min) / accuracy + 1)
    weight_num = int((weight_max - weight_min) / weight_acc + 1)
    b_off = b_true
    a_off = a_true
    s_linspace = np.sqrt(np.linspace(s_min, s_max, grid_num))
    weight_linspace = np.linspace(weight_min, weight_max, weight_num)
    output = TS_rep_multiprocess_s(total_run, t_off, t_on, s_linspace, b_off, a_off, weight_linspace)
    filename = filehead + '1_' + str(t_off) + '_' + str(t_on) + '.mat'
    savemat(filename, mdict={md: output})
    output = MP_rep_multiprocess_s(total_run, t_off, t_on, s_linspace, b_off, a_off, weight_linspace)
    filename = filehead + '2_' + str(t_off) + '_' + str(t_on) + '.mat'
    savemat(filename, mdict={md: output})
    output = CVP_rep_multiprocess_s(total_run, t_off, t_on, s_linspace, b_off, a_off, weight_linspace)
    filename = filehead + '3_' + str(t_off) + '_' + str(t_on) + '.mat'
    savemat(filename, mdict={md: output})
    output = MCS_rep_multiprocess_s(total_run, t_off, t_on, s_linspace, b_off, a_off, weight_linspace)
    filename = filehead + '4_' + str(t_off) + '_' + str(t_on) + '.mat'
    savemat(filename, mdict={md: output})
if __name__ == "__main__":
    main()
