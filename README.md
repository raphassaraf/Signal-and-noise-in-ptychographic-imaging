# Signal and noise in ptychographic imaging
![initial_object_probe](https://github.com/user-attachments/assets/ea938258-faeb-4b78-8798-015eb7ac5fbd)

## Contributors
* Raphael Assaraf - raphael.assaraf@epfl.ch

## Introduction
This project contains all the code used to generate the results and plots of the _Signal and noise in ptychographic imaging_ white paper.
The work was done in the scope of a semester project at the Computational X-ray Imaging Group at EPFL, in collaboration with the 
Paul Scherrer Institute.

## Project Structure

* `/figures/*`     -  All saved figures as .png files
* `/logs/*`     -  An empty folder where all logs file are sent during simulations
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

### Usage
The simulations should be run from terminal as:
```bash
sbatch <simulation_code.sh>
```

The results are then saved in the simulation's corresponding folder under the *outputs/* directory.

Once the simulations have been run, the results can be plotted with the *analysis.ipynb* file.


## Acknowledgements

Parts of the project were inspired by and use the CDTools library from Abe Levitan: https://cdtools-developers.github.io/cdtools-docs/examples.html
