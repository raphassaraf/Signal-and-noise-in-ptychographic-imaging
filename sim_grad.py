import numpy as np
import torch as t

from rec_scripts import reconstruct_flu
from utils import save_output

multiprobe = 'grad'
w_start, w_end, w_step = [100, 0, 0], [0, 50, 50], 11
weights_scan = np.linspace(w_start, w_end, w_step)

flu_i, flu_f, flu_n = -1, 6, 20
fluences = np.logspace(flu_i, flu_f, flu_n)

print(
    f'''Begin simulations with:
        fluences: logspace({flu_i, flu_f, flu_n}),
        multiprobe: {multiprobe} with linspace({w_start, w_end, w_step})'''
)

weights_dict_keys = [
    np.array2string(weights, separator=',').replace(' ', '').replace('.', '')
    for weights in weights_scan
    ]
rec_dict_keys = ['object_mse', 'loss_mse', 'object_mse_nll', 'loss_mse_nll']

rec_dict = {
    rec_key: {w_key: None for w_key in weights_dict_keys} for rec_key in rec_dict_keys
}

### MSE ###
for weights, weights_key in zip(weights_scan, weights_dict_keys):
    print(f'\nWeights = {weights_key}\n')
    (
        rec_dict['object_mse'][weights_key],
        rec_dict['loss_mse'][weights_key]
    ) = reconstruct_flu(fluences, 'MSE',
                        multiprobe='grad', grad_weights=weights)
    save_output(rec_dict, 'rec_grad.pkl')

### MSE --> NLL ###
for weights, weights_key in zip(weights_scan, weights_dict_keys):
    print(f'\nWeights = {weights_key}\n')
    (
        rec_dict['object_mse_nll'][weights_key],
        rec_dict['loss_mse_nll'][weights_key]
    ) = reconstruct_flu(fluences, 'MSE', 'PoissonNLL',
                        multiprobe='grad', grad_weights=weights)
    
    save_output(rec_dict, 'rec_grad.pkl')
