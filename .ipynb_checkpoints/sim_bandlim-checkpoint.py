import argparse
import numpy as np

from tools.rec_scripts import reconstruct_flu
from tools.utils import *


def main():

    parser = argparse.ArgumentParser(
        prog='sim_bandlim',
        description=(
            'Run a reconstruction sweep over different amounts of '
            'band-limited random modes. For each amount of modes, '
            'sweep over different fluences.'
        )
    )
    parser.add_argument(
        'n_modes',
        type=int,
        help='The number of modes to simulate.'
    )
    parser.add_argument(
        'n_fluences',
        type=int,
        help='The number of fluences between 1e-1 and 1e6 to simulate.'
    )
    args = parser.parse_args()

    n_modes = args.n_modes
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
        multiprobe='band_lim',
        n_probes=n_modes
    )
    save_output(rec_dict, f'outputs/bandlim/rec_bandlim_{n_modes}.pkl')
        
    ### MSE --> PNLL ###
    rec_dict['object_mse_nll'], rec_dict['loss_mse_nll'] = reconstruct_flu(
        fluences,
        'MSE',
        'PoissonNLL',
        multiprobe='band_lim',
        n_probes=n_modes
    )
            
    save_output(rec_dict, f'outputs/bandlim/rec_bandlim_{n_modes}.pkl')


if __name__ == '__main__':

    main()
    




