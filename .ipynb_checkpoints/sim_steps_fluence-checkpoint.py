import numpy as np
import torch as t

from rec_scripts import reconstruct_steps_flu
from utils import save_output

flu_i, flu_f, flu_n = -1, 6, 30
fluences = np.logspace(flu_i, flu_f, flu_n)

st_i, st_f, st_f = 1, 31, 1
steps_sizes = np.arange(st_i, st_f, st_f)

print(
    f'''Begin simulations with:
        fluences: logspace({flu_i, flu_f, flu_n})
        steps_sizes: arange({st_i, st_f, st_f})'''
)

rec_dict = {
    'object_mse': None,
    'loss_mse': None,
    'object_mse_nll': None,
    'loss_mse_nll': None
}

### MSE ###
rec_dict['object_mse'], rec_dict['loss_mse'] = reconstruct_steps_flu(
    fluences, steps_sizes, 'MSE'
)
save_output(rec_dict, 'rec_steps_fluence_test.pkl')

### MSE --> NLL ###
rec_dict['object_mse_nll'], rec_dict['loss_mse_nll'] = reconstruct_steps_flu(
    fluences, steps_sizes, 'MSE', 'PoissonNLL'
)
save_output(rec_dict, 'rec_steps_fluence_test.pkl')
