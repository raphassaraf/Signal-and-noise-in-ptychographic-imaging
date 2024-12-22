# Signal and noise in ptychographic imaging
### Stochastic Simulations - Assignment 1
![initial_object_probe](https://github.com/user-attachments/assets/ea938258-faeb-4b78-8798-015eb7ac5fbd)


### Contributors
* Raphael Assaraf - raphael.assaraf@epfl.ch

## Table of Contents

* [Introduction](#introduction)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Licensing](#license)
* [Acknowledgements](#acknowledgements)

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
    - `./utils.py` - contains the main area estimation class, `Mandelbrot`.
* `main.py`      - 
* `requirements.txt` -


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

## Usage of CLA's

The simulations can be run and saved via the terminal.

Structure of command line argument:
```bash
python3 main.py method [--help] [-n N_samples] [-i ITERATIONS] [-s SIMULATIONS] [--symmetry] [--stratified] [--save]
```
`method` is the only mandatory argument and must be one of the following: [random, hypercube, orthogonal]

`-n` is used to pass the number of samples, must have an integer square root (default is 2500).

`-i` is the precision used when checking for convergence (default is 500 iterations).

`-s` number of simulations to run (default is 10).

`--symmetry` to exploit symmetry in the x-axis when estimating the area.

`--stratified` to enable stratified sampling (to reduce variance)

`--save` saves results in a csv-file in `./data/`.



### Example usage
Running and saving 10 simulations with stratified random sampling with 2500 samples and iteration limit of 500
```bash
python3 main.py random -n 2500 -i 500 -s 10 --save
```
After executing the simulations, main.py will print the mean area and the corresponding sample variance.
```
Area found using random sampling: 1.4918592
Variance: 0.0018081583103999972
```

Running and saving 50 simulations of stratified hypercube sampling with 4096 samples and iteration limit of 1000:
```bash
python3 main.py hypercube -n 4096 -i 1000 -s 10 --save
```

For a summary on the usage of main.py and its commandline arguments, run:
```bash
python3 main.py -h
```
## Licensing
This project is licensed under the [MIT License](LICENSE.md) - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

Code for the Mandlebrot visuals and Julia set images were adapted from [this report](https://medium.com/swlh/visualizing-the-mandelbrot-set-using-python-50-lines-f6aa5a05cf0f).
This code was accessed via its public repository on github: [Mandelbrot-Set](https://github.com/blakesanie/Mandelbrot-Set)
