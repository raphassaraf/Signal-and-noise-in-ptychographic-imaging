import itertools
import numpy as np
import torch as t

from rec_scripts import reconstruct_flu
from utils import save_output

multiprobe = 'defocus'
distances_set = [5, 25, 45, 65, 85, 105]
distances_scan = np.array(list((itertools.combinations(distances_set, 4))))

flu_i, flu_f, flu_n = -1, 6, 30
fluences = np.logspace(flu_i, flu_f, flu_n)

print(
    f'''Begin simulations with:
        fluences: logspace({flu_i, flu_f, flu_n}),
        multiprobe: {multiprobe} with combinations({distances_set, 4})'''
)

dist_dict_keys = [
    np.array2string(distances, separator=',').replace(' ', '')
    for distances in distances_scan
    ]
rec_dict_keys = ['object_mse', 'loss_mse', 'object_mse_nll', 'loss_mse_nll']
rec_dict = {
    rec_key: {d_key: None for d_key in dist_dict_keys} for rec_key in rec_dict_keys
}

### MSE ###
for distances, d_key in zip(distances_scan, dist_dict_keys):
    print(f'\Distances = {d_key}\n')
    (
        rec_dict['object_mse'][d_key],
        rec_dict['loss_mse'][d_key]
    ) = reconstruct_flu(fluences, 'MSE',
                        multiprobe='defocus', propa_distances=distances)
    save_output(rec_dict, 'rec_defocus.pkl')

### MSE --> NLL ###
for distances, d_key in zip(distances_scan, dist_dict_keys):
    print(f'\Distances = {d_key}\n')
    (
        rec_dict['object_mse_nll'][d_key],
        rec_dict['loss_mse_nll'][d_key]
    ) = reconstruct_flu(fluences, 'MSE', 'PoissonNLL',
                        multiprobe='defocus', propa_distances=distances)
    
    save_output(rec_dict, 'rec_defocus.pkl')
