import argparse
import numpy as np

from tools.rec_scripts import reconstruct_flu
from tools.utils import *


def main():

    parser = argparse.ArgumentParser(
        prog='steps_bandlim_reconstruction',
        description=(
            'Run a reconstruction sweep over scanning step size going from 1 to 30.'
            'Use a 20 pixelwide band-limited random probe.'
            'For each step size, reconstructions are performed over different fluences.'
        )
    )
    parser.add_argument(
        'steps_size',
        type=int,
        help=(
            'The scanning steps size.'
        )
    )
    parser.add_argument(
        'n_fluences',
        type=int,
        help='The number of fluences between 1e-1 and 1e6 to simulate.'
    )
    args = parser.parse_args()

    steps_size = args.steps_size
    print(f'Steps size: {steps_size}')
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
        steps_size=steps_size,
        multiprobe='band_lim',
        n_probes=1
    )
    save_output(rec_dict, f'outputs/steps_bandlim5/rec_steps_bandlim5_{steps_size}.pkl')
            
    ### MSE --> PNLL ###
    rec_dict['object_mse_nll'], rec_dict['loss_mse_nll'] = reconstruct_flu(
        fluences,
        'MSE',
        'PoissonNLL',
        steps_size=steps_size,
        multiprobe='band_lim',
        n_probes=1
    )
                
    save_output(rec_dict, f'outputs/steps_bandlim5/rec_steps_bandlim5_{steps_size}.pkl')


if __name__ == '__main__':

    main()
