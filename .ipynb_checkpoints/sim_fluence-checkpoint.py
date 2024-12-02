import numpy as np
import torch as t

from rec_scripts import reconstruct_flu
from utils import save_output

flu_i, flu_f, flu_n = -1, 6, 40
fluences = np.logspace(flu_i, flu_f, flu_n)

print(
    f'''Begin simulations with:
        fluences: logspace({flu_i, flu_f, flu_n})'''
)

rec_dict = {
    'object_mse': None,
    'loss_mse': None,
    'object_mse_nll': None,
    'loss_mse_nll': None
}

### MSE ###
rec_dict['object_mse'], rec_dict['loss_mse'] = reconstruct_flu(
    fluences, 'MSE'
)
save_output(rec_dict, 'rec_fluence.pkl')

### MSE --> NLL ###
rec_dict['object_mse_nll'], rec_dict['loss_mse_nll'] = reconstruct_flu(
    fluences, 'MSE', 'PoissonNLL'
)
save_output(rec_dict, 'rec_fluence.pkl')
