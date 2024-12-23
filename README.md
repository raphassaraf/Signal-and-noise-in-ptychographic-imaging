# Signal and noise in ptychographic imaging
![initial_object_probe](figures/initial_object_probe.png)

## Contributors
* Raphael Assaraf - raphael.assaraf@epfl.ch

## Introduction
This project contains all the code used to generate the results and plots of the _Signal and noise in ptychographic imaging_ white paper.
The work was done in the scope of a semester project at the Computational X-ray Imaging Group at the Swiss Federal Institute of Technology of Lausanne (EPFL), in collaboration with the 
Paul Scherrer Institute.

## Project Structure

* `/figures/*`     -  All saved figures as .png files
* `/logs/*`     -  Folder where all logs file are sent during simulations
* `/outputs/*`     - Contains all the outputs of the simulations as .pkl files
    - `./bandlim/*` - Outputs of the simulations with multi-probes based on band-limited random modes 
    - `./grad_1direction/*` - Outputs of the simulations with multi-probes based on directional derivatives 
    - `./steps_bandlim5/*` - Outputs of the varying step sizes simulations
    - `./rec_fluence.pkl` - Outputs of the initial fluence sweep
* `/tools/*`     -  Contains the functions and constants used in the simulations
    - `./constants.py` - Global variables used in the project
    - `./metrics.py` - Functions used for calculating the error metrics
    - `./rec_scripts.py` - Reconstruction function called in the .sh simulation scripts
    - `./tools_plt.py` - Functions for plotting results
    - `./tools_rec.py` - Ptychography and image processing related functions
    - `./utils.py` - Utility functions 
* `analysis.ipynb` - The notebook for plotting the results
* `sim_bandlim.sh` - Reconstructions with increasing amount of band-limited random modes, over varying fluences
* `sim_fluence.sh` - Reconstructions with varying fluences using de-focused ring probe
* `sim_grad_1direction.sh` - Reconstructions with varying modes intensities for de-focused ring + y-directional derivative modes, over varying fluences
* `sim_steps_bandlim.sh` - Reconstructions with varying step sizes using band-limited random probe, over varying fluences
* `requirements.txt` - All required dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/raphassaraf/Signal-and-noise-in-ptychographic-imaging.git
```
2. Navigate to the project root directory.
```
cd ./Signal-and-noise-in-ptychographic-imaging/
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
The simulations should be run using Slurm, from the terminal as:
```bash
sbatch {simulation_code.sh}
```

The results are then saved in the simulation's corresponding folder under the *outputs/* directory.

Once the simulations have been run, the results can be plotted with the *analysis.ipynb* file.


## Acknowledgements

Parts of the project were inspired by and/or use the [CDTools]([url](https://cdtools-developers.github.io/cdtools-docs/examples.html)) library from Abraham Levitan.
