import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series (sinusoidal with noise)
torch.manual_seed(42)
T = 200  # Total time steps
t = torch.linspace(0, 20, T)
y = torch.sin(2 * torch.pi * t / 5) + 0.1 * torch.randn(T)  # Noisy periodic signal

# Simulate a system change (e.g., frequency change at T=150)
y[150:] = torch.sin(2 * torch.pi * t[150:] / 3.5) + 0.1 * torch.randn(T-150)  

# Create lagged dataset
lag = 5
X = torch.stack([y[i:i+lag] for i in range(T-lag)])
Y = y[lag:]  # The next value to predict

# Split data into training (before change) and testing (after change)
X_train, Y_train = X[:140], Y[:140]  
X_test, Y_test = X[140:], Y[140:]  
print("Training data shape: ", X_train.shape, Y_train.shape)
# Define Gaussian Process Model with Spectral Mixture Kernel
class SpectralMixtureGP(gpytorch.models.ExactGP):
    def __init__(self, X_train, Y_train, likelihood, num_mixtures=2):
        super(SpectralMixtureGP, self).__init__(X_train, Y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.cov_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=lag)
        self.cov_module.initialize_from_data(X_train, Y_train)

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.cov_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

# Initialize model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGP(X_train, Y_train, likelihood)

# Train model
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter=4000
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, Y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()

# Make predictions and detect anomalies
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred_dist = model(X_test)
    Y_pred = pred_dist.mean
    Y_std = pred_dist.stddev

# Compute confidence intervals
upper_bound = Y_pred + 2 * Y_std
lower_bound = Y_pred - 2 * Y_std

# Detect anomalies where observed value is outside the confidence interval
anomalies = (Y_test > upper_bound) | (Y_test < lower_bound)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(len(Y)), Y.numpy(), 'k', label="True Values")
plt.plot(range(len(Y_train)), Y_train.numpy(), 'b', label="Training Data")
plt.plot(range(len(Y_train), len(Y)), Y_pred.numpy(), 'r', label="GP Prediction")
plt.fill_between(range(len(Y_train), len(Y)), lower_bound.numpy(), upper_bound.numpy(), color='r', alpha=0.2, label="Confidence Interval")
plt.scatter(np.where(anomalies.numpy())[0] + len(Y_train), Y_test[anomalies].numpy(), color='red', marker='x', label="Detected Change")
plt.legend()
plt.title("Time Series Change Detection with Spectral Mixture Kernel")
plt.show()
