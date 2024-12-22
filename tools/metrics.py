import numpy as np
import torch as t

from skimage.metrics import structural_similarity


def nmse_phase(pred_img, true_img):
    '''Compute the normalised mean square error between two complex images.
    
    Args:
        pred_img: 2-dimensional complex tensor of the reconstructed object
        true_img: 2-dimensional complex tensor of the true object
        
    Returns:
        The normalised mean square error'''
    
    mse = (
        (t.abs(true_img)**2).sum()
        + (t.abs(pred_img)**2).sum()
        - 2*(t.abs((t.conj(true_img)*pred_img).sum()))
    )
    
    return mse / (t.abs(true_img)**2).sum()


def get_MSE_flu(true_img, rec_obj_dict):
    '''Compute the normalised mean square errors between two complex images, for each fluence.
    
    Args:
        true_img: 2-dimensional complex tensor of the true object
        rec_obj_dict: dictionnary with the keys being the fluences
          and values being the corresponding reconstructed objects
        
    Returns:
        The array containing the normalised mean square errors for each fluence'''

    fluences = np.array(list(rec_obj_dict.keys()))

    mse_array = np.zeros(fluences.shape[0])
    for i, (flu, pred_img) in enumerate(rec_obj_dict.items()):
        mse_array[i] = nmse_phase(pred_img, true_img).item()
            
    return mse_array


def get_SSIM_flu(true_img, rec_obj_dict):
    '''Compute the structural similarity index measures between two complex images, for each fluence.
    
    Args:
        true_img: 2-dimensional complex tensor of the true object
        rec_obj_dict: dictionnary with the keys being the fluences
          and values being the corresponding reconstructed objects
        
    Returns:
        The array containing the structural similarity index measures for each fluence'''

    fluences = np.array(list(rec_obj_dict.keys()))

    ssim_array = np.zeros(fluences.shape[0])
    for i, (flu, pred_img) in enumerate(rec_obj_dict.items()):
        ssim_array[i] = structural_similarity(
            t.abs(pred_img).numpy(),
            t.abs(true_img).numpy(),
            data_range=(
                t.abs(true_img).max() - t.abs(true_img).min()
            ).item()
        )
            
    return ssim_array


def get_MSE_param_flu(true_img, rec_obj_dict):
    '''Compute the normalised mean square error between two complex images, for each fluence
    and each variation of an experimental parameter (step size or multi-probe configuration)
    
    Args:
        true_img: 2-dimensional complex tensor of the true object
        rec_obj_dict: dictionnary in the form {parameter: {fluence: reconstructed_object}}
        
    Returns:
        The (n_parameter, n_fluence) shaped array containing the normalised mean square errors 
        for each fluence and each parameter'''

    params = np.array(list(rec_obj_dict.keys()))
    fluences = np.array(list(rec_obj_dict[params[0]].keys()))

    mse_array = np.zeros((params.shape[0], fluences.shape[0]))
    
    for i, (p, fluence_dict) in enumerate(rec_obj_dict.items()):
        for j, (fluence, pred_img) in enumerate(fluence_dict.items()):
            mse_array[i][j] = nmse_phase(pred_img, true_img).item()
            
    return mse_array


def get_SSIM_param_flu(true_img, rec_obj_dict):
    '''Compute the normalised structural similarity index measures between two complex images, for each fluence
    and each variation of an experimental parameter (step size or multi-probe configuration)
    
    Args:
        true_img: 2-dimensional complex tensor of the true object
        rec_obj_dict: dictionnary in the form {parameter: {fluence: reconstructed_object}}
        
    Returns:
        The (n_parameter, n_fluence) shaped array containing the structural similarity index measures 
        for each fluence and each parameter'''

    params = np.array(list(rec_obj_dict.keys()))
    fluences = np.array(list(rec_obj_dict[params[0]].keys()))

    ssim_array = np.zeros((params.shape[0], fluences.shape[0]))
    
    for i, (p, fluence_dict) in enumerate(rec_obj_dict.items()):
        for j, (fluence, pred_img) in enumerate(fluence_dict.items()):
            ssim_array[i][j] = structural_similarity(
                t.abs(pred_img).numpy(),
                t.abs(true_img).numpy(),
                data_range=(
                    t.abs(true_img).max() - t.abs(true_img).min()
                ).item()
            )
            
    return ssim_array