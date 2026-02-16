import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)         
plt.rc('axes', titlesize=BIGGER_SIZE)    
plt.rc('axes', labelsize=BIGGER_SIZE)    
plt.rc('xtick', labelsize=SMALL_SIZE)    
plt.rc('ytick', labelsize=SMALL_SIZE)    
plt.rc('legend', fontsize=MEDIUM_SIZE)  
plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('figure', dpi=300)


def plot_metrics_flu(fluences, mse_dict, ssim_dict):
    '''
    Plot the mse and 1-ssim metrics over each simulated fluence.

    Args:
        fluences: np.array of fluences
        mse_dict: dictionnary containing the results of the mean square error
        ssim_dict: dictionnary containing the results of the structural similarity index
    
    Returns:
        The figure object containing the subplots'''

    fig, ax = plt.subplots(1, 2, figsize=(8, 2.5))

    legends = ['MSE', 'MSE - PNLL', 'old MSE']
    colors = ['b', 'r']

    for mse, ssim, l, c in zip(mse_dict.values(), ssim_dict.values(), legends, colors):
        ax[0].scatter(fluences, mse, s=5, label=l, c=c)
        ax[1].scatter(fluences, 1 - ssim, s=5, label=l, c=c)
        
    for a in ax: a.grid(), a.set_xlabel('fluence'), a.legend(), a.set_xscale('log'), a.set_yscale('log')
    ax[0].set_ylabel('mse')
    ax[1].set_ylabel('1-ssim')
    fig.tight_layout()

    return fig


def plot_metrics_params_flu(fluences, mse_dict, ssim_dict, param, param_keys, idx_param=None):
    '''
    Plot the mse and 1-ssim metrics over each simulated fluence and variation of the specified experimental parameter.

    Args:
        fluences: np.array of fluences
        mse_dict: dictionnary containing the results of the mean square error
        ssim_dict: dictionnary containing the results of the structural similarity index
        param: {'steps', 'grad', 'defocus, 'band_lim'} - the parameter that was varied on in the simulations
        param_keys: np.array of strings containing the keys representing each variation of the experimental parameter
        idx_param: np.array containing of the indices to the keys to consider (ex: if only step sizes 3, 7 and 11 should be plotted)
    
    Returns:
        The figure object containing the subplots'''

    if param == 'steps': unit = 'pixels'
    elif param == 'grad': unit = '%'
    elif param == 'defocus': unit = 'm'
    elif param == 'band_lim': unit = 'modes'

    if idx_param is None: idx_param = np.arange(0, len(param_keys))

    f, ax = plt.subplots(2, 2, figsize=(8, 5))
    colormap = plt.cm.viridis
    colors = np.array([colormap(i) for i in np.linspace(1, 0, idx_param.shape[0])])
    mark_size = 4

    for i, (mse_array, ssim_array) in enumerate(zip(mse_dict.values(), ssim_dict.values())):
        for s, mse, ssim, clr in zip(
            param_keys[idx_param], mse_array[idx_param],
            ssim_array[idx_param], colors
        ):
            if s == 1 and (param == 'steps' or param == 'band_lim'):
                ax[0, i].loglog(fluences, mse, label=f'{s} {unit}'[:-1], c=clr, linewidth=.5, marker='o', markersize=mark_size)
            elif param == 'grad':
                ax[0, i].loglog(fluences, mse, label=f'{s}'[:-3]+f'] {unit}', c=clr, linewidth=.5, marker='o', markersize=mark_size)
            else:
                ax[0, i].loglog(fluences, mse, label=f'{s} {unit}', c=clr, linewidth=.5, marker='o', markersize=mark_size)
            ax[1, i].loglog(fluences, 1 - ssim, c = clr, linewidth=.5, marker='o', markersize=mark_size)

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