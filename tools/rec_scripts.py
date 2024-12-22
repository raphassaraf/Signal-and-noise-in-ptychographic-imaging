import numpy as np
import torch as t

from tools.constants import PATH, OBJ_SIZE
from tools.tools_rec import *


DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

##### Reconstruction script for raster scan on fluence #####
def reconstruct_flu(fluences, loss_fct_1, loss_fct_2=None, probe_r_ratio=5, steps_size=1, 
                    multiprobe=None, propa_distances=None, grad_weights=None, n_probes=None):
    '''Performs object reconstructions with known probes at difference fluences.
    
    Args:
        fluences: array of fluences to sweep on
        loss_fct_1: the first (or only) loss function to use for the reconstructions
        loss_fct_2: the second loss function to use for the reconstructions
        probe_r_ratio: int, ratio between the size of a side and the radius of the outer ring for the de-focused ring probe
        steps_size: int, number of pixels between each scanning position
        multiprobe: the multiprobe generation method {'defocus', 'grad', 'band_lim'}
        propa_distances: int, the propagation distances to use in case of 'defocus' multiprobe simulation
        grad_weights: (3,) shaped array, the weights for each mode in case of 'grad' multiprobe simulation
        n_probes: int, number of modes to simulate in case of 'band_lim' multiprobe simulation
        
    Returns:
        reconstructed_obj_dict: dictionnary containing the reconstructed object for each fluence
        reconstructed_loss_dict: dictionnary containing the loss history for each fluence'''

    print(f'Tensors initialization on device {DEVICE}')
    sim_obj = simulate_object(PATH, resize=OBJ_SIZE, bc='periodic').to(DEVICE)

    print(f'Multiprobe = {multiprobe}')
    if multiprobe == 'defocus':
        sim_probe = simulate_multiprobe_defocus(
            OBJ_SIZE, r_ratio=probe_r_ratio, propagate=True,
            propa_distances=propa_distances
        )
    elif multiprobe == 'grad':
        sim_probe = simulate_multiprobe_grad(
            OBJ_SIZE, r_ratio=probe_r_ratio, propagate=True,
            distance=60, weights=grad_weights
        )
    elif multiprobe == 'band_lim':
        sim_probe = simulate_multiprobe_band_limited_random(
            OBJ_SIZE, r_ratio=probe_r_ratio, n_probes=n_probes
        )
        print(sim_probe.shape)
    else:
        sim_probe = simulate_probe(
            OBJ_SIZE, r_ratio=probe_r_ratio, propagate=True, distance=60
        )
    sim_probe = sim_probe.to(DEVICE)

    translations = set_scanning_grid(
        coord = (OBJ_SIZE, OBJ_SIZE),
        n_steps = np.ceil(OBJ_SIZE/steps_size),
        steps_size = steps_size,
        add_noise = False
    ).to(DEVICE)

    # Dictionnary containing the diffraction patterns and rescaled probe for each
    # fluence.
    true_diff_probe_dict = {
        f: get_diffractions_fluence(
            sim_probe, sim_obj, translations, fluence=f, noise='poisson'
        )
        for f in fluences
    }

    # Dictionnaries to store each reconstructed object and losses per fluences.
    reconstructed_obj_dict = dict.fromkeys(fluences)
    reconstructed_loss_dict = dict.fromkeys(fluences)

    if loss_fct_2 is None:
        epochs_1 = 100
    else:
        epochs_1 = 15

    ##### Reconstruction loop #####
    print(f'\n{loss_fct_1, loss_fct_2} RECONSTRUCTIONS\n')
    for flu, diff_probe in true_diff_probe_dict.items():
        
        true_diff = diff_probe[0].to(DEVICE) # actual diffraction patterns tensor
        true_probe = diff_probe[1].to(DEVICE) # rescaled probe tensor
        obj_guess = get_obj_guess(OBJ_SIZE).to(DEVICE)
            
        print(f'Reconstruction for fluence = {flu} photons/pixel')
        obj, loss_1 = AD_model_LBFGS(
            diffractions = true_diff,
            obj_guess = obj_guess,
            sim_probe = true_probe,
            translations = translations,
            epochs = epochs_1,
            loss_f=loss_fct_1,
        )

        if loss_fct_2 is not None:
            obj, loss_2 = AD_model_LBFGS(
                diffractions = true_diff,
                obj_guess = obj_guess,
                sim_probe = true_probe,
                translations = translations,
                epochs = 100,
                loss_f=loss_fct_2,
            )
            reconstructed_loss_dict[flu] =  {loss_fct_1: loss_1, loss_fct_2: loss_2}
                
        else: 
            reconstructed_loss_dict[flu] =  {loss_fct_1: loss_1}
            
        
        reconstructed_obj_dict[flu] = obj

    return reconstructed_obj_dict, reconstructed_loss_dict