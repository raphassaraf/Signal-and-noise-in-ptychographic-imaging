#!/usr/bin/env python3
"""
Unified experiment runner for ptychographic imaging experiments.

This module provides an interface to easily run all 4 reconstruction experiments:
  1. sim_bandlim: Sweep over band-limited random modes and fluences
  2. sim_fluence: Sweep over fluences
  3. sim_grad_1direction: Sweep over gradient mode weights and fluences
  4. sim_steps_bandlim: Sweep over scanning step sizes and fluences
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add experiments directory to path
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))


def run_bandlim_experiment(n_modes, n_fluences):
    """
    Run the band-limited modes experiment.
    
    Args:
        n_modes: Number of band-limited modes to simulate
        n_fluences: Number of fluences to simulate
    """
    print(f"\n{'='*70}")
    print(f"Running BANDLIM experiment: n_modes={n_modes}, n_fluences={n_fluences}")
    print(f"{'='*70}\n")
    
    from experiments import sim_bandlim
    sys.argv = ['sim_bandlim.py', str(n_modes), str(n_fluences)]
    sim_bandlim.main()


def run_fluence_experiment(n_fluences):
    """
    Run the fluence sweep experiment.
    
    Args:
        n_fluences: Number of fluences to simulate
    """
    print(f"\n{'='*70}")
    print(f"Running FLUENCE experiment: n_fluences={n_fluences}")
    print(f"{'='*70}\n")
    
    from experiments import sim_fluence
    sys.argv = ['sim_fluence.py', str(n_fluences)]
    sim_fluence.main()


def run_grad_1direction_experiment(principal_mode_weight, n_fluences):
    """
    Run the gradient mode (1 direction) experiment.
    
    Args:
        principal_mode_weight: Weight index for principal mode
        n_fluences: Number of fluences to simulate
    """
    print(f"\n{'='*70}")
    print(f"Running GRAD_1DIRECTION experiment: principal_mode_weight={principal_mode_weight}, n_fluences={n_fluences}")
    print(f"{'='*70}\n")
    
    from experiments import sim_grad_1direction
    sys.argv = ['sim_grad_1direction.py', str(principal_mode_weight), str(n_fluences)]
    sim_grad_1direction.main()


def run_steps_bandlim_experiment(steps_size, n_fluences):
    """
    Run the scanning step size sweep experiment.
    
    Args:
        steps_size: Scanning step size
        n_fluences: Number of fluences to simulate
    """
    print(f"\n{'='*70}")
    print(f"Running STEPS_BANDLIM experiment: steps_size={steps_size}, n_fluences={n_fluences}")
    print(f"{'='*70}\n")
    
    from experiments import sim_steps_bandlim
    sys.argv = ['sim_steps_bandlim.py', str(steps_size), str(n_fluences)]
    sim_steps_bandlim.main()


def main():
    parser = argparse.ArgumentParser(
        prog='run_experiment',
        description='Unified runner for ptychographic imaging experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run individual experiments
  python run_experiment.py bandlim --n-modes 5 --n-fluences 20
  python run_experiment.py fluence --n-fluences 50
  python run_experiment.py grad-1direction --principal-mode-weight 5 --n-fluences 20
  python run_experiment.py steps-bandlim --steps-size 10 --n-fluences 20
  
  # Run all experiments with default parameters
  python run_experiment.py all
        """
    )
    
    subparsers = parser.add_subparsers(dest='experiment', help='Experiment to run')
    
    # Bandlim experiment
    bandlim_parser = subparsers.add_parser(
        'bandlim',
        help='Band-limited modes sweep experiment'
    )
    bandlim_parser.add_argument(
        '--n-modes',
        type=int,
        default=5,
        help='Number of band-limited modes (default: 5)'
    )
    bandlim_parser.add_argument(
        '--n-fluences',
        type=int,
        default=20,
        help='Number of fluences (default: 20)'
    )
    
    # Fluence experiment
    fluence_parser = subparsers.add_parser(
        'fluence',
        help='Fluence sweep experiment'
    )
    fluence_parser.add_argument(
        '--n-fluences',
        type=int,
        default=50,
        help='Number of fluences (default: 50)'
    )
    
    # Gradient 1 direction experiment
    grad_parser = subparsers.add_parser(
        'grad-1direction',
        help='Gradient mode (1 direction) experiment'
    )
    grad_parser.add_argument(
        '--principal-mode-weight',
        type=int,
        default=5,
        help='Weight index for principal mode (default: 5)'
    )
    grad_parser.add_argument(
        '--n-fluences',
        type=int,
        default=20,
        help='Number of fluences (default: 20)'
    )
    
    # Steps bandlim experiment
    steps_parser = subparsers.add_parser(
        'steps-bandlim',
        help='Scanning step size sweep experiment'
    )
    steps_parser.add_argument(
        '--steps-size',
        type=int,
        default=10,
        help='Scanning step size (default: 10)'
    )
    steps_parser.add_argument(
        '--n-fluences',
        type=int,
        default=20,
        help='Number of fluences (default: 20)'
    )
    
    # All experiments
    all_parser = subparsers.add_parser(
        'all',
        help='Run all experiments with default parameters'
    )
    all_parser.add_argument(
        '--n-fluences',
        type=int,
        default=20,
        help='Number of fluences for all experiments (default: 20)'
    )
    
    args = parser.parse_args()
    
    if args.experiment == 'bandlim':
        run_bandlim_experiment(args.n_modes, args.n_fluences)
    elif args.experiment == 'fluence':
        run_fluence_experiment(args.n_fluences)
    elif args.experiment == 'grad-1direction':
        run_grad_1direction_experiment(args.principal_mode_weight, args.n_fluences)
    elif args.experiment == 'steps-bandlim':
        run_steps_bandlim_experiment(args.steps_size, args.n_fluences)
    elif args.experiment == 'all':
        run_bandlim_experiment(5, args.n_fluences)
        run_fluence_experiment(args.n_fluences)
        run_grad_1direction_experiment(5, args.n_fluences)
        run_steps_bandlim_experiment(10, args.n_fluences)
        print(f"\n{'='*70}")
        print("All experiments completed successfully!")
        print(f"{'='*70}\n")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
