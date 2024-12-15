import matplotlib.pyplot as plt
import numpy as np
import torch as t

from matplotlib.ticker import FormatStrFormatter


plt.rcParams.update({
    'font.size': 12,
    'xtick.labelsize' : 12,
    'ytick.labelsize' : 12,
    'axes.labelsize' : 12,
    'legend.fontsize': 12,
    'lines.linewidth' : 1,
    'figure.dpi' : 100
})


def plot_amplitude(img, ax):
    
    ax.imshow(t.abs(img))
    
    pass


def plot_phase(img, ax):
    
    ax.imshow(t.angle(img))
    
    pass


def plot_loss(loss, ax):

    keys = list(loss.keys())

    if len(keys) == 1:
        ax.plot(loss[keys[0]])
        
    elif len(keys) == 2:
        c1 = 'tab:blue'
        c2 = 'tab:orange'
        
        ax.plot(loss[keys[0]], c=c1)
        ax.set_ylabel(keys[0], c=c1)
        ax.tick_params(axis='y', labelcolor=c1)

        ax2 = ax.twinx()
        ax2.plot(loss[keys[1]], c=c2)
        ax2.set_ylabel(keys[1], c=c2)
        ax2.tick_params(axis='y', labelcolor=c2)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    pass


def plot_results_flu(
    rec_obj_dict, rec_loss_dict, indices
):

    fluences = np.array(list(rec_obj_dict.keys()))

    # selection of the dictionnary elements within the fluences slice. 
    obj_dict = {key: rec_obj_dict[key] for key in fluences[indices]}
    loss_dict = {key: rec_loss_dict[key] for key in fluences[indices]}
    
    f, ax = plt.subplots(3, len(indices), figsize=(4*len(indices), 9))
    for i, ((fluence, img), loss) in enumerate(zip(obj_dict.items(), loss_dict.values())):
        plot_amplitude(img, ax[0, i]) # Reconstructed object amplitude.
        plot_phase(img, ax[1, i]) # Reconstructed object phase.
        plot_loss(loss, ax[2, i]) # Loss evolution.
        ax[0, i].set_title(f'{fluence:.2f} photons/pixel')
        
    ax[0, 0].set_ylabel('Amplitude')
    ax[1, 0].set_ylabel('Angle')
    ax[2, 0].set_ylabel('Loss')
    f.tight_layout()

    return f


def plot_metric_flu(fluences, mse_dict, ssim_dict):
    f, ax = plt.subplots(1, 2, figsize=(8, 3))

    legends = ['MSE', 'MSE - PNLL', 'old MSE']

    for mse, ssim, l in zip(mse_dict.values(), ssim_dict.values(), legends):
        ax[0].scatter(fluences, mse, s=5, label=l)
        ax[1].scatter(fluences, 1 - ssim, s=5, label=l)
        
    for a in ax: a.grid(), a.set_xlabel('fluence'), a.legend(), a.set_xscale('log'), a.set_yscale('log')
    ax[0].set_ylabel('mse')
    ax[1].set_ylabel('1-ssim')
    f.tight_layout()

    return f


def plot_results_param_flu(result_dict, plot_fct, idx_param, idx_flu, param):

    if param == 'steps': unit = 'pixels'
    elif param == 'grad': unit = '%'
    elif param == 'defocus': unit = 'm'
    elif param == 'band_lim': unit = 'modes'

    params = np.array(list(result_dict.keys()))[idx_param]
    fluences = np.array(list(result_dict[params[0]].keys()))[idx_flu]
    
    new_dict = {
        p: {flu: result_dict[p][flu] for flu in fluences} for p in params
    }

    figsize = (4*fluences.shape[0], 3*params.shape[0])
    f, ax = plt.subplots(len(params), len(fluences), figsize=figsize)
    
    if len(params) == 1: ax = np.expand_dims(ax, axis=0)
    
    for i, (p, fluence_dict) in enumerate(new_dict.items()):
        for j, (fluence, data) in enumerate(fluence_dict.items()):
            plot_fct(data, ax[i, j])
        ax[i, 0].set_ylabel(f'{p} {unit}')
        
    for i, fluence in enumerate(fluences): 
        ax[0, i].set_title(f'{fluence:.2f} photons/pixel')

    ax = ax.flatten()
    if plot_fct != plot_loss:
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

    f.tight_layout()

    return f


def plot_metrics_params_flu(fluences, mse_dict, ssim_dict, param, param_keys, idx_param=None):

    if param == 'steps': unit = 'pixels'
    elif param == 'grad': unit = '%'
    elif param == 'defocus': unit = 'm'
    elif param == 'band_lim': unit = 'modes'

    if idx_param is None: idx_param = np.arange(0, len(param_keys))

    f, ax = plt.subplots(2, 2, figsize=(8, 6))
    colormap = plt.cm.brg
    colors = np.array([colormap(i) for i in np.linspace(1, 0, idx_param.shape[0])])

    for i, (mse_array, ssim_array) in enumerate(zip(mse_dict.values(), ssim_dict.values())):
        for s, mse, ssim, clr in zip(
            param_keys[idx_param], mse_array[idx_param],
            ssim_array[idx_param], colors
        ):
            ax[0, i].loglog(fluences, mse, label=f'{s} {unit}', c=clr, linewidth=.5)
            ax[1, i].loglog(fluences, 1 - ssim, c = clr, linewidth=.5)

    ax[0, 0].set_title('MSE reconstruction')
    ax[0, 1].set_title('MSE - PNLL reconstruction')
    ax[0, 0].set_ylabel('mse')
    ax[1, 0].set_ylabel('1-ssim')
    ax[1, 0].set_xlabel('fluence'), ax[1, 1].set_xlabel('fluence')
    ax = ax.flatten()
    for a in ax: a.grid(),
    handles, labels = ax[0].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper center', ncol=1, bbox_to_anchor=[-0.07, 0.96])
            
    f.tight_layout()
    
    return f