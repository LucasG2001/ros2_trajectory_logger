import numpy as np
import GPy
import matplotlib.pyplot as plt
from filters import compute_length_scale_from_fft, compute_length_scale_from_fft_windowed

def fit_gaussian_gp(data: np.ndarray, n_window: int = 25, optimize: bool = True) -> GPy.models.GPRegression:
    """
    Fits a standard Gaussian Process (GP) regression model to predict a signal using its shifted version.
    
    Parameters:
        data (np.ndarray): 1D array of the signal data.
        n_window (int): Number of time steps to shift the signal. Defaults to 25.
        optimize (bool): Whether to optimize the GP hyperparameters. Defaults to True. If False the hyperparams are set via FFT analysis. 
        
    Returns:
        model (GPy.models.GPRegression): Trained GP model.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    
    N = len(data)
    print("length of data is ", N)
    # Pad the signal with zeros at the beginning
    padded_data = np.concatenate((np.zeros(n_window), data))
    # Sample every 100th point in the data
    sampled_indices = np.arange(n_window + 1, len(padded_data), 100)
    sampled_data = padded_data[sampled_indices]
    # Create input-output pairs for training
    X_test = np.array([padded_data[i - n_window - 1: i -1] for i in range(n_window + 1, len(padded_data))]) 
    X = np.array([padded_data[i - n_window - 1: i -1] for i in sampled_indices])
    print("sampled {} points".format(len(X)))
    Y = sampled_data.reshape(-1, 1)  # Target values
    print("length of Y is ", len(Y))
    
    # Reshape X to match expected input format for GPy
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    
    # Define and fit the GP model
    if optimize:
         kernel = GPy.kern.RBF(input_dim=n_window)
         model = GPy.models.GPRegression(X, Y, kernel)
    else:
        # Compute length scale from FFT
        l = compute_length_scale_from_fft_windowed(data, sc=5, fs=500, plot=True)
        l = np.sqrt(l)
        kernel = GPy.kern.RBF(input_dim=n_window, variance=1.0, lengthscale=l)
        model = GPy.models.GPRegression(X, Y, kernel, noise_var=0.03)
        # Constrain lengthscale between 0.5*l and 1.5*l
        model.kern.lengthscale.constrain_bounded(0.5 * l, 1.5 * l)

    #optimize the model
    model.optimize()

    # Print optimized model parameters
    print("Optimized model parameters:")
    print(model)
    # Plot the fitted model
    Y_pred, Y_var = model.predict(X_test)
    Y_std = np.sqrt(Y_var)
    
    plt.figure(figsize=(10, 5))
    plt.plot(data, '--b', label='True Signal')
    plt.plot(Y_pred, 'r', label='Predicted Signal')
    plt.fill_between(range(N-1), Y_pred.flatten() - 2 * Y_std.flatten(),
                     Y_pred.flatten() + 2 * Y_std.flatten(), color='r', alpha=0.2)
    
    # Plot the sampled points
    plt.scatter(sampled_indices - n_window, sampled_data, marker='o', color='k', label='Sampled Points')
    
    plt.legend()
    plt.title("Gaussian Process Regression with Sampled Points")
    plt.show()
    
    return model
