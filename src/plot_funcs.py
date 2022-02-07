import numpy as np
import matplotlib.pyplot as plt

from helper_funcs import *

# Args for plotting
class PlotArguments:
    log_filename = "benchmark_v2.csv"
    columns = ['tef', 'method', 'trial', 'safety', 'discomfort', 'time']
    columns_to_normalize = ['safety', 'discomfort', 'time']

def parse_log_columns(plot_args, pre_normalize=False):
    logs = read_log_from_file(plot_args.log_filename)
    res = {v: (logs[:,i]).astype('float64') if can_be_float(logs[0,i]) else (logs[:,i]).astype('str') for (i,v) in enumerate(plot_args.columns)}
    if not pre_normalize:
        return res
    else:
        return normalize_columns(res, plot_args)

def get_scores_min_max(y, yerr, keys):
    min_vals = np.array([y[k] - yerr[k] for k in keys])
    max_vals = np.array([y[k] + yerr[k] for k in keys])
    return np.min(min_vals, axis=(0,1)), np.max(max_vals, axis=(0,1))

def normalize_columns(res, plot_args):
    for c in plot_args.columns_to_normalize:
        res[c] = normalize_from_0_to_1(res[c])
    return res

def get_mean_and_ste(logs_dict, idx):
    vals = np.array([logs_dict['safety'][idx], logs_dict['discomfort'][idx], logs_dict['time'][idx]])
    avg = np.mean(vals, axis=1)
    ste = np.std(vals, axis=1) / np.sqrt(vals.shape[-1])
    return avg, ste

plot_args = PlotArguments()
logs_dict = parse_log_columns(plot_args)
tef_values = np.unique(logs_dict['tef'])
method_values = np.unique(logs_dict['method'])
len_cols_to_norm = len(plot_args.columns_to_normalize)

# Pre-allocate `y` and its error values `yerr`
plot_y_vals = {m:np.zeros((len(tef_values), len_cols_to_norm)) for m in method_values}
plot_yerr_vals = {m:np.zeros((len(tef_values), len_cols_to_norm)) for m in method_values}

for (a, tef) in enumerate(tef_values):
    idx_tef = np.squeeze(np.argwhere(logs_dict['tef']==tef))

    for method in method_values:
        idx_method = np.squeeze(np.argwhere(logs_dict['method']==method))

        idx = np.intersect1d(idx_tef, idx_method)
        y, yerr = get_mean_and_ste(logs_dict, idx)
        plot_y_vals[method][a, :] = y
        plot_yerr_vals[method][a, :] = yerr
        

plot_min, plot_max = get_scores_min_max(plot_y_vals, plot_yerr_vals, method_values)
fig, subplt = plt.subplots(len_cols_to_norm, 1)
fig.suptitle('Scores vs. Training Effort')
x = tef_values

for (c, score) in enumerate(plot_args.columns_to_normalize):
    for method in method_values:
        y = (plot_y_vals[method][:,c]) / plot_max[c]
        yerr = plot_yerr_vals[method][:,c] / plot_max[c]
        subplt[c].errorbar(x, y, yerr=yerr, label=f"{method}")
        subplt[c].set_ylabel(score)
        subplt[c].legend(loc='upper left')

subplt[c].set_xlabel("training effort")
plt.show()

