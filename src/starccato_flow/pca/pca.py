from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
This module implements PCA-based Bayesian parameter estimation for gravitational wave signals.
It includes functions for computing the Advanced LIGO PSD, empirical spectrum, colored noise generation,
Fourier array conversion, and Bayesian PCR reconstruction.
'''

def AdvLIGOPsd(f):
    x = f / 215
    x2 = x * x
    psd = 1e-49 * (pow(x, - 4.14) - 5 / x2 + 111 * (1 - x2 + 0.5 * x2 * x2) / (1 + 0.5 * x2))
    # The upper bound is 2e10 times the minimum value
    cutoff = np.nanmin(psd) * 2e10
    psd[(psd > cutoff) | np.isnan(psd)] = cutoff
    return psd

def FreqArray(signal, sampling_freq = 4096):
    # Convert to even for symmetry
    N = len(signal) // 2 * 2
    # Sampling interval
    delta_t = 1 / sampling_freq
    # Frequency interval
    delta_f = sampling_freq / N
    # Fourier frequencies
    f = np.arange(N // 2 + 1) * delta_f
    k = np.array([0] + [1] * (N // 2 - 1) + [0])
    return N, delta_t, f, k

# Returns the one-sided empirical spectrum of a time series
def EmpiricalSpectrum(signal):
    # Get Fourier frequencies array
    N, delta_t, _, k = FreqArray(signal)
    # Non-redundant Fourier transform
    FT = np.fft.fft(signal)[:(N // 2 + 1)]
    # Common coefficients
    coef = (k + 1) * np.sqrt(delta_t / N)
    # Cosine coefficients
    a = coef * FT.real
    # Sine coefficients
    b = - coef * FT.imag
    # Spectral power based on Fourier transform
    power = (a ** 2 + b ** 2) / (k + 1)
    return power

# Given a one-sided PSD, generate colored noise in the time domain
def NoiseGenerator(df_signal, x = 1):
    def ColoredNoise(signal):
        # Get Fourier frequencies array
        N, delta_t, f, k = FreqArray(signal)
        # SD of Fourier frequencies
        sigma_f = np.sqrt(AdvLIGOPsd(f) / pow(k + 1, 2))
        # Sample of normal random variable
        a = np.random.normal(loc = 0, scale = sigma_f, size = N // 2 + 1)
        b = np.random.normal(loc = 0, scale = sigma_f, size = N // 2 + 1) * k
        # Find real and imaginary parts
        real = np.sqrt(N / delta_t) * a
        real = np.concatenate([real, np.flip(real[k == 1])])
        imag = np.sqrt(N / delta_t) * b
        imag = np.concatenate([- imag, np.flip(imag[k == 1])])
        # Complex noise vector in frequency domain
        noiseFT = real + 1j * imag
        # Inverse FT to TS and take real parts
        noiseTS = np.fft.ifft(noiseFT).real
        return noiseTS
    # Generate noise based on the input signal
    noise = df_signal.apply(ColoredNoise) * x
    # Mean Centering
    noise = noise - noise.mean()
    return noise

def FourierArray(signal):
    # Get Fourier frequencies array
    N, _, _, k = FreqArray(signal)
    # Non-redundant Fourier transform
    FT = np.fft.fft(signal)[:(N // 2 + 1)]
    FA = np.concatenate([FT.real, FT.imag[k == 1]])
    return FA

def BayesianPCR(X, y_train = signal_Tran, t = times, n = 20, n_samples = 10000):
    # Select the first 20 principal components
    pca = PCA(n_components = n)
    # Generate principal component matrix based on signals (y_train)
    PCx = pd.DataFrame(pca.fit_transform(y_train), columns = [f'PC{i + 1}' for i in range (n)])
    # Generate Fourier array for principal component matrix
    PCx_FT = PCx.apply(FourierArray)
    # Generate Fourier array for signals with noise
    X_FT = X.reset_index(drop = True).apply(FourierArray)
    # Get Fourier frequencies array
    N, delta_t, f, k = FreqArray(X)
    # Noise spectral density
    PSD = AdvLIGOPsd(f)
    # The standard deviation vector corresponding to PCx_FT and X
    SD = np.sqrt(N / delta_t * np.concatenate([PSD / pow(k + 1, 2), PSD[k == 1] / pow(k[k == 1] + 1, 2)]))
    # Create the diagonal inverse-variance matrix
    Dinv = np.diag(1 / SD ** 2)
    # Posterior covariance regression coefficients
    Sigma = np.linalg.pinv(PCx_FT.T @ Dinv @ PCx_FT)
    # Posterior mean regression coefficients
    Mu = Sigma @ PCx_FT.T @ Dinv @ X_FT
    # Reconstruct signal
    def Reconstruction(Mu, Sigma = Sigma):
        # Direct Sampling from posterior distribution (N * 20)
        Post_samples = np.random.multivariate_normal(mean = Mu, cov = Sigma, size = n_samples)
        # 10000 samples for each time point (256 * N)
        Sign_samples = np.dot(PCx, Post_samples.T)
        # Return the mean of each time point (256)
        return np.mean(Sign_samples, axis = 1)
    # Return the signal (y_pred)
    Recon_signal = Mu.apply(Reconstruction)
    Recon_signal.index = t['time'].to_numpy()
    return Recon_signal
