#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:51:30 2024

@author: yuewang
"""

import numpy as np

a_min = 8.0
a_max = 10.0
b_min = 4.0
b_max = 5.0
p_min = 0.7
p_max = 1.25
a_true = 9.0
b_true = 4.5
noise_true = 0.1
price_true = a_true / (2 * b_true)
mesh = 0.05
a_len = int((a_max - a_min) / mesh + 1)
b_len = int((b_max - b_min) / mesh + 1)
a = np.arange(a_len)[:, None] * mesh + a_min
b = np.arange(b_len)[None, :] * mesh + b_min

def log_gauss_density(x, mu, var):
    den = np.exp(- (x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    return np.log(den)

def generate_offline_data_matrix(t_off, a_off, b_off, noise_off):
    grid = (p_max - p_min) / (t_off - 1)
    data_off_p = np.arange(t_off) * grid + p_min
    data_off_d = a_off - (data_off_p) * b_off + np.random.normal(loc=0.0, scale=noise_off, size=t_off)
    return data_off_p, data_off_d

def posterior_offline_matrix(data_off_p, data_off_d, t_off, a_off, b_off, noise_off, weight_off):
    curr_noise = data_off_d[:, None, None] - a[None, ...] + b[None, ...] * data_off_p[:, None, None]
    log_dist = (weight_off * log_gauss_density(curr_noise.reshape(t_off, -1), 0, noise_true)).sum(axis=0)

    log_dist -= max(log_dist)
    curr_dist = np.exp(log_dist)
    curr_dist /= sum(curr_dist)
    return log_dist, curr_dist

def posterior_online_matrix(log_dist, curr_dist, price_chosen, demand_chosen):
    curr_noise = demand_chosen - a + b * price_chosen
    log_dist += log_gauss_density(curr_noise.flatten(), 0, noise_true)
    log_dist -= max(log_dist)
    curr_dist = np.exp(log_dist)
    curr_dist /= sum(curr_dist)
    return log_dist, curr_dist


##########################

def TS(t_off, t_on, a_off, b_off, noise_off, weight_off):
    data_off_p, data_off_d = generate_offline_data_matrix(t_off, a_off, b_off, noise_off)
    log_dist, curr_dist = posterior_offline_matrix(data_off_p, data_off_d, t_off, a_off, b_off, noise_off, weight_off)
    total_regret = 0.0
    for i in range(t_on):
        para_chosen = np.random.choice(range(a_len * b_len), size=1, p=curr_dist)[0]
        a_chosen = (para_chosen // b_len) * mesh + a_min
        b_chosen = (para_chosen % b_len) * mesh + b_min
        price_chosen = a_chosen / (2 * b_chosen)
        total_regret += price_true * (a_true - b_true * price_true) \
            - price_chosen * (a_true - b_true * price_chosen)
        demand_chosen = a_true - b_true * price_chosen \
            + np.random.normal(loc=0.0, scale=noise_true)
        log_dist, curr_dist = posterior_online_matrix(log_dist, curr_dist, price_chosen, demand_chosen)
    return total_regret

def MP(t_off, t_on, a_off, b_off, noise_off, weight_off):  
    data_off_p, data_off_d = generate_offline_data_matrix(t_off, a_off, b_off, noise_off)
    log_dist, curr_dist = posterior_offline_matrix(data_off_p, data_off_d, t_off, a_off, b_off, noise_off, weight_off)
    total_regret = 0.0
    for i in range(t_on):
        para_chosen = np.argmax(curr_dist)
        a_chosen = (para_chosen // b_len) * mesh + a_min
        b_chosen = (para_chosen % b_len) * mesh + b_min
        price_chosen = a_chosen / (2 * b_chosen)
        total_regret += price_true * (a_true - b_true * price_true) \
            - price_chosen * (a_true - b_true * price_chosen)
        demand_chosen = a_true - b_true * price_chosen \
            + np.random.normal(loc=0.0, scale=noise_true)
        log_dist, curr_dist = posterior_online_matrix(log_dist, curr_dist, price_chosen, demand_chosen)
    return total_regret

def CVP(t_off, t_on, a_off, b_off, noise_off, weight_off):  
    alpha = 0.5001
    data_off_p, data_off_d = generate_offline_data_matrix(t_off, a_off, b_off, noise_off)
    log_dist, curr_dist = posterior_offline_matrix(data_off_p, data_off_d, t_off, a_off, b_off, noise_off, weight_off)
    total_regret = 0.0
    for t in range(t_on):
        para_chosen = np.argmax(curr_dist)
        a_chosen = (para_chosen // b_len) * mesh + a_min
        b_chosen = (para_chosen % b_len) * mesh + b_min
        p_curr = a_chosen / (2 * b_chosen)
        data_off_p = np.append(data_off_p, p_curr)
        if len(data_off_p) < 2:
            c = 0
        else:
            c = 2 ** (-alpha) * (data_off_p[0] - data_off_p[1]) ** 2 * min(1, 1/3/alpha) / 2
        if np.var(data_off_p) < c * (t_off + t + 1) ** (alpha - 1):
            p_l = np.mean(data_off_p[:-1]) - \
                np.sqrt(c * ((t_off + t + 1) ** alpha - \
                (t_off + t)**alpha) * ((t_off + t + 1) / (t_off + t)))
            p_h = np.mean(data_off_p[:-1]) + \
                np.sqrt(c * ((t_off + t + 1) ** alpha - \
                (t_off + t)**alpha) * ((t_off + t + 1) / (t_off + t)))
            if p_l * (a_chosen - b_chosen * p_l) < p_h * (a_chosen - b_chosen * p_h):
                price_chosen = p_h
            else:
                price_chosen = p_l
        else:
            price_chosen = p_curr    
        data_off_p[-1] = price_chosen
        total_regret += price_true * (a_true - b_true * price_true) \
            - price_chosen * (a_true - b_true * price_chosen)
        demand_chosen = a_true - b_true * price_chosen \
            + np.random.normal(loc=0.0, scale=noise_true)
        log_dist, curr_dist = posterior_online_matrix(log_dist, curr_dist, price_chosen, demand_chosen)
    return total_regret


def MCS(t_off, t_on, a_off, b_off, noise_off, weight_off):  
    data_off_p, data_off_d = generate_offline_data_matrix(t_off, a_off, b_off, noise_off)
    log_dist, curr_dist = posterior_offline_matrix(data_off_p, data_off_d, t_off, a_off, b_off, noise_off, weight_off)
    total_regret = 0.0
    total_stage = 0
    K = 2
    p_explore = np.array([])
    grid = (p_max - p_min) / (K - 1)    
    for k in range(K):
        p_exp = p_min + k * grid
        p_explore = np.append(p_explore, p_exp)
    c = 0   
    while total_stage < t_on:        
        c += 1
        if total_stage + K + c > t_on:
            n_periods = t_on - (total_stage)
        elif total_stage + K + c <= t_on:
            n_periods = K + c        
        for s in range(n_periods):
            # exploration phase
            if s <= K - 1:
                price_chosen = p_explore[s]
                demand_chosen = a_true - b_true * price_chosen + np.random.normal(loc=0.0, scale=noise_true)
                log_dist, curr_dist = posterior_online_matrix(log_dist, curr_dist, price_chosen, demand_chosen)
            if s == K - 1:
                para_curr = np.argmax(curr_dist)
                a_chosen = (para_curr // b_len) * mesh + a_min
                b_chosen = (para_curr % b_len) * mesh + b_min            
                price_curr = a_chosen / (2 * b_chosen)
            # exploitation phase
            if s > K - 1:               
                price_chosen = price_curr
                demand_chosen = a_true - b_true * price_chosen + np.random.normal(loc=0.0, scale=noise_true)
                log_dist, curr_dist = posterior_online_matrix(log_dist, curr_dist, price_chosen, demand_chosen)            
            total_regret += price_true * (a_true - b_true * price_true) \
                            - price_chosen * (a_true - b_true * price_chosen)   
        total_stage += n_periods 
    return total_regret




