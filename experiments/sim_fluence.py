import argparse
import numpy as np

from src.reconstruction import reconstruct_flu
from src.utils import *


def main():

    parser = argparse.ArgumentParser(
        prog='steps_size_reconstruction',
        description=(
            'Run a reconstruction sweep over multiple fluences.'
        )
    )
    parser.add_argument(
        'n_fluences',
        type=int,
        help='The number of fluences between 1e-1 and 1e6 to simulate.'
    )
    args = parser.parse_args()

    n_flu = args.n_fluences
    flu_i, flu_f = -2, 8
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
    )
    save_output(rec_dict, f'outputs/rec_fluence.pkl')
        
    ### MSE --> PNLL ###
    rec_dict['object_mse_nll'], rec_dict['loss_mse_nll'] = reconstruct_flu(
        fluences,
        'MSE',
        'PoissonNLL',
    )
            
    save_output(rec_dict, f'outputs/rec_fluence.pkl')


if __name__ == '__main__':

    main()

