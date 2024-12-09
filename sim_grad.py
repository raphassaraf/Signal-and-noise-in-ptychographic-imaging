import argparse
import itertools
import numpy as np
import torch as t

from tools.rec_scripts import reconstruct_flu
from tools.utils import *


def main():

    parser = argparse.ArgumentParser(
        prog='sim_grad',
        description=(
            'Run a reconstruction sweep over the following multiprobe system: '
            '[w_1*P, w_2*Px, w_3*Py]. Px and Py are the x and y directional '
            'derivatives of P, and the weights w_i vary from [100, 0, 0] to '
            '[0, 50, 50] by decreasing w_1 by 10, and increasing w_2 and w_3 '
            'by 5 at each iteration. For each weight configuration, sweep over '
            'different fluences.'
        )
    )
    parser.add_argument(
        'principal_mode_weight',
        type=int,
        help=(
            'The weight attributed to the principal mode P'
            '(the x and y derivatives will have equally distributed weights '
            'w_2 and w_3, such that the total distributed weights equal to 100).'
        )
    )
    parser.add_argument(
        'n_fluences',
        type=int,
        help='The number of fluences between 1e-1 and 1e6 to simulate.'
    )
    args = parser.parse_args()

    principal_mode_weight_list = np.arange(0, 101, 10)
    principal_mode_weight = principal_mode_weight_list[-args.principal_mode_weight]
    grad_modes_weight = (100 - principal_mode_weight) / 2
    weights = np.array([principal_mode_weight, grad_modes_weight, grad_modes_weight])
    pkl_suf = np.array2string(weights, separator=',').replace(' ', '').replace('.', '')
    print(f'Weights: {weights}')

    n_flu = args.n_fluences
    flu_i, flu_f = -1, 6
    fluences = np.logspace(flu_i, flu_f, n_flu)
    
    rec_dict = {
        'object_mse': None,
        'loss_mse': None,
        'object_mse_nll': None,
        'loss_mse_nll': None
    }
    
    ### MSE ###
    rec_dict['object_mse'], rec_dict['loss_mse'] = reconstruct_flu(
        fluences,
        'MSE',
        multiprobe='grad',
        grad_weights=weights
    )
    save_output(rec_dict, f'outputs/grad/rec_grad_{pkl_suf}.pkl')
        
    ### MSE --> NLL ###
    rec_dict['object_mse_nll'], rec_dict['loss_mse_nll'] = reconstruct_flu(
        fluences,
        'MSE',
        'PoissonNLL',
        multiprobe='grad',
        grad_weights=weights
    )
            
    save_output(rec_dict, f'outputs/grad/rec_grad_{pkl_suf}.pkl')


if __name__ == '__main__':

    main()





