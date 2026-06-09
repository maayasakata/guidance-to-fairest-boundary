import numpy as np
import math
import torch
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import yaml
from scipy.special import expit 

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)

def cal_thresh_logit(logit0, logit1, delta, epsilon=1e-5):
    from metrics.ddp import DDP
    p1 = len(logit1) / (len(logit0) + len(logit1))
    p0 = 1 - p1
    const = 10
    
    z_max0 = (max(logit0) + const)
    z_max1 = (max(logit1) + const)
    z_min0 = (min(logit0) - const)
    z_min1 = (min(logit1) - const)
    upper_z0 = p0 * -np.tanh(z_min0/2)
    upper_z1 = p1 * np.tanh(z_max1/2)
    lower_z0 = p0 * -np.tanh(z_max0/2)
    lower_z1 = p1 * np.tanh(z_min1/2)

    ddp = DDP(logit0, logit1, tau0=0, tau1=0)

    if abs(ddp) <= delta:
        t_star = 0
        log0 = log1 = 0

    elif ddp > delta: # t > 0
        t_min = 0
        t_max = max(upper_z0, upper_z1)
        # bisection search
        while t_max - t_min > epsilon:
            t = (t_max + t_min) / 2 
            inner_log0 = (p0+t)/(p0-t)
            inner_log1 = (p1-t)/(p1+t)
            log0 = -math.log(inner_log0) if inner_log0 > 0 else z_min0 
            log1 = -math.log(inner_log1) if inner_log1 > 0 else z_max1
            tau0 = log0
            tau1 = log1
            ddp = DDP(logit0, logit1, tau0=tau0, tau1=tau1)
            if ddp > delta:
                t_min = t
            else: 
                t_max = t
        t_star = (t_min + t_max) / 2
        inner_log0 = (p0+t_star)/(p0-t_star)
        inner_log1 = (p1-t_star)/(p1+t_star)
        log0 = -math.log(inner_log0) if inner_log0 > 0 else z_min0 
        log1 = -math.log(inner_log1) if inner_log1 > 0 else z_max1 

    else:  
        t_min = min(lower_z0, lower_z1)
        t_max = 0
        # bisection search
        while t_max - t_min > epsilon:
            t = (t_max + t_min) / 2 
            inner_log0 = (p0+t)/(p0-t)
            inner_log1 = (p1-t)/(p1+t)
            log0 = -math.log(inner_log0) if inner_log0 > 0 else z_max0 
            log1 = -math.log(inner_log1) if inner_log1 > 0 else z_min1 
            tau0 = log0
            tau1 = log1
            ddp = DDP(logit0, logit1, tau0=tau0, tau1=tau1)
            if ddp < -delta:
                t_max = t
            else: 
                t_min = t
        t_star = (t_min + t_max) / 2
        inner_log0 = (p0+t_star)/(p0-t_star)
        inner_log1 = (p1-t_star)/(p1+t_star)
        log0 = -math.log(inner_log0) if inner_log0 > 0 else z_max0 
        log1 = -math.log(inner_log1) if inner_log1 > 0 else z_max1 

    tau0_star = log0
    tau1_star = log1
    ddp = DDP(logit0, logit1, tau0_star, tau1_star)

    return tau0_star, tau1_star, t_star, ddp

def generate_tradeoff(delta_list, eta0, eta1, eta0_te, eta1_te, y0, y1):
    from metrics.ddp import DDP
    from metrics.acc import ACC
    current_res = []
    for delta in delta_list:
        tau0, tau1, _, _ = cal_thresh_logit(eta0, eta1, delta)
        ddp = DDP(eta0_te, eta1_te, tau0, tau1)
        acc0 = ACC(eta0_te, y0, tau0)
        acc1 = ACC(eta1_te, y1, tau1)
        acc = (len(y0)*acc0 + len(y1)*acc1) / (len(y0) + len(y1))
        current_res.append({'delta': delta, 'acc': acc, 'ddp': abs(ddp)})
    return pd.DataFrame(current_res)

def make_fig(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].scatter(df["acc"], df["ddp"], color='blue', label='DDP', marker='o')
    axes[0].set_title('Average DDP vs Accuracy')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('DDP')
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].scatter(df["acc"], df["delta"], color='orange', label='Accuracy', marker='o')
    axes[1].set_title('Average Accuracy per Delta')
    axes[1].set_xlabel('Accuracy')
    axes[1].set_ylabel('Delta')
    axes[1].grid(True)
    axes[1].legend()
    axes[1].axis('equal') 

    plt.tight_layout()
    plt.close()
    return fig

def cal_avg_curve(delta_list, seeds, eta0s, eta1s, eta0s_test, eta1s_test, y0s, y1s):
    from metrics.Hypervolume import Hypervolume
    seed_results = []
    hv_list = []
    for i in range(seeds):
        result = generate_tradeoff(delta_list, eta0s[i], eta1s[i], eta0s_test[i], eta1s_test[i], y0s[i], y1s[i])
        result["seed"] = i*5
        hv = Hypervolume(result)
        hv_list.append(hv)
        seed_results.append(result)

    all_seeds_df = pd.concat(seed_results, ignore_index=True)
    df_test = all_seeds_df.groupby('delta').agg({
                'acc': 'mean',
                'ddp': 'mean'
            }).reset_index()
    result_all = pd.DataFrame(df_test)
    hv_var = np.std(hv_list)
    hv_mean = np.mean(hv_list)
    return result_all, hv_mean, hv_var, seed_results
