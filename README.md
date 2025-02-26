# ros2_trajectory_logging

# Python Scripts
filters.py - filter functions used throughout the package. Implements fourier-spectrum based hyperparameter estimation for Gaussian processes.
hyperparams_from_fourier.py - implements an example to compare hyperparameter tuning based on frequency spectrum and marginal likelihood maximization
gaussian_on_slices.py - implements static GP regression on a dataset or slice of it
GP_regression.py - implements various functions to simulate a RT GP-fitting process. Main testing script.
helpers.py - self-explanatory
RTBandPassFilter.py - implements a Real-time band-pass filter as Class
SpikeDetector.py - Class containing functionality to store and operate on loaded data
tests.py - example functions to test filters in "filters.py"