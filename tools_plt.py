import matplotlib.pyplot as plt
import numpy as np
import torch as t

from matplotlib.ticker import FormatStrFormatter


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
    
    f, ax = plt.subplots(3, len(indices), figsize=(20, 10))
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


def plot_results_param_flu(result_dict, plot_fct, idx_param, idx_flu, param):

    if param == 'steps': unit = 'pixels'
    elif param == 'grad': unit = '%'
    elif param == 'defocus': unit = 'm'

    params = np.array(list(result_dict.keys()))[idx_param]
    fluences = np.array(list(result_dict[params[0]].keys()))[idx_flu]
    
    new_dict = {
        p: {flu: result_dict[p][flu] for flu in fluences} for p in params
    }
    
    f, ax = plt.subplots(len(params), len(fluences), figsize=(20, 20))
    
    if len(params) == 1: ax = np.expand_dims(ax, axis=0)
    
    for i, (p, fluence_dict) in enumerate(new_dict.items()):
        for j, (fluence, data) in enumerate(fluence_dict.items()):
            plot_fct(data, ax[i, j])
        ax[i, 0].set_ylabel(f'{p} {unit}')
        
    for i, fluence in enumerate(fluences): 
        ax[0, i].set_title(f'{fluence:.2f} photons/pixel')
    
    f.tight_layout()

    return f