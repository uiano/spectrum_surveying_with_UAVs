# Spectrum Surveying:

The Python code in this repository implements the simulations and plots the figures described in the paper “Spectrum Surveying: Active Radio Map Estimation with Autonomous UAVs” by Raju Shrestha, Daniel Romero, and Sundeep Prabhakar Chepuri.

# Requirements: 
Python 3.6 or later. 
Use the package manager pip to install the following packages:
```
tensorflow 2.4.0
tensorflow-probability 0.12.1 
scipy  1.5.2
cvxpy 1.1.7
matplotlib 3.3.2
pandas 1.2.0 
joblib 1.0.1 
sklearn 0.23.2
opencv-python 4.5.1.48
dill 0.3.3
seaborn 0.11.1 
```


# Guidelines

First, download all the files and folders from this repository. Then, the simulations can be executed by running the file `run_experiment.py`from the command prompt. 
One needs to provide the experiment number (e.g. 2001) as an argument while executing the file `run_experiment.py`to select the simulation you want to run. 
The experiments reproducing different figures in the paper are organized in the methods located in the file `Experiments/spectrum_surveying_experiments.py`.
The comments before each method indicate which figure(s) on the paper it generates.
For example, to run the experiment no 2001 in the `Experiments/spectrum_surveying_experiments.py`, in the command prompt, execute the command `$ python run_experiment.py 2001`.

For more information about the simulation environment, please check [here](https://github.com/fachu000/GSim-Python).


# Citation
If our code is helpful in your research or work, please cite our paper: “Spectrum Surveying: Active Radio Map Estimation with Autonomous UAVs.”


#### Contact
Please feel free to contact us by email if you have any issues in running the code.
