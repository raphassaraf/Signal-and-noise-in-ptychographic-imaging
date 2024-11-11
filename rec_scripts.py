import numpy as np
import torch as t

from constants import PATH, OBJ_SIZE
from tools_rec import *


DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

##### Reconstruction script for raster scan on fluence #####
def reconstruct_flu(fluences, loss_fct_1, loss_fct_2=None, multiprobe=None,
                    propa_distances=None, grad_weights=None):

    print(f'Tensors initialization on device {DEVICE}')
    sim_obj = simulate_object(PATH, resize=OBJ_SIZE, bc='periodic').to(DEVICE)

    print(f'Multiprobe = {multiprobe}')
    if multiprobe == 'defocus':
        sim_probe = simulate_multiprobe_defocus(
            OBJ_SIZE, r_ratio=5, propagate=True,
            propa_distances=propa_distances
        )
    elif multiprobe == 'grad':
        sim_probe = simulate_multiprobe_grad(
            OBJ_SIZE, r_ratio=5, propagate=True,
            distance=60, weights=grad_weights
        )
    else:
        sim_probe = simulate_probe(
            OBJ_SIZE, r_ratio=5, propagate=True, distance=60
        )
    sim_probe = sim_probe.to(DEVICE)

    translations = set_scanning_grid(
        coord = (OBJ_SIZE, OBJ_SIZE),
        n_steps = OBJ_SIZE,
        steps_size = 1,
        add_noise = False
    ).to(DEVICE)

    # Dictionnary containing the diffraction patterns and rescaled probe for each
    # fluence.
    true_diff_dict = {
        f: get_diffractions_fluence(
            sim_probe, sim_obj, translations, fluence=f, noise='poisson'
        )
        for f in fluences
    }

    # Dictionnaries to store each reconstructed object and losses per fluences.
    reconstructed_obj_dict = dict.fromkeys(fluences)
    reconstructed_loss_dict = dict.fromkeys(fluences)

    if loss_fct_2 is None:
        epochs_1 = 1000
    else:
        epochs_1 = 15

    ##### Reconstruction loop #####
    print(f'\n{loss_fct_1, loss_fct_2} RECONSTRUCTIONS\n')
    for flu, diff_patterns in true_diff_dict.items():
    
        true_diff = diff_patterns[0].to(DEVICE) # actual diffraction patterns tensor
        true_probe = diff_patterns[1].to(DEVICE) # rescaled probe tensor
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
                epochs = 1000,
                loss_f=loss_fct_2,
            )
            reconstructed_loss_dict[flu] =  {loss_fct_1: loss_1, loss_fct_2: loss_2}
                
        else: 
            reconstructed_loss_dict[flu] =  {loss_fct_1: loss_1}
            
        
        reconstructed_obj_dict[flu] = obj

    return reconstructed_obj_dict, reconstructed_loss_dict


##### Reconstruction script for raster scan on steps and fluence #####
def reconstruct_steps_flu(fluences, steps_sizes, loss_fct_1, loss_fct_2=None): # loss_fct_1, loss_fct_2 = {'MSE', 'PoissonNLL'}
    
    print(f'Tensors initialization on device {DEVICE}')
    sim_obj = simulate_object(PATH, resize=OBJ_SIZE, bc='periodic').to(DEVICE)
    sim_probe = simulate_probe(OBJ_SIZE, r_ratio=5, propagate=True, distance=60).to(DEVICE)

    # Dictionnaries to store the results (reconstructed object and loss values).
    reconstructed_obj_dict = {
        size: {f: None for f in fluences} for size in steps_sizes
    }
    reconstructed_loss_dict = {
        size: {f: None for f in fluences} for size in steps_sizes
    }

    if loss_fct_2 is None:
        epochs_1 = 1000
    else:
        epochs_1 = 15

    ##### Reconstruction loop #####
    print(f'\n{loss_fct_1, loss_fct_2} RECONSTRUCTIONS\n')
    for size in steps_sizes:
        
        translations = set_scanning_grid(
            coord = (OBJ_SIZE, OBJ_SIZE),
            n_steps = np.ceil(OBJ_SIZE/size),
            steps_size = size,
            add_noise = False
        ).to(DEVICE)
        
        for flu in fluences:
            
            true_diff, true_probe = get_diffractions_fluence(
                sim_probe, sim_obj, translations, noise='poisson', fluence=flu
            )
            true_diff = true_diff.to(DEVICE)
            true_probe = true_probe.to(DEVICE)
            obj_guess = get_obj_guess(OBJ_SIZE).to(DEVICE)
        
            print(f'Reconstruction for (step_size, fluence) = ({size}, {flu})')
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
                    epochs = 1000,
                    loss_f=loss_fct_2,
                )
                reconstructed_loss_dict[size][flu] =  {loss_fct_1: loss_1, loss_fct_2: loss_2}
                
            else: 
                reconstructed_loss_dict[size][flu] =  {loss_fct_1: loss_1}
            
            reconstructed_obj_dict[size][flu] = obj

    return reconstructed_obj_dict, reconstructed_loss_dict