import numpy as np

SIGNAL_COLOUR = "deepskyblue"
GENERATED_SIGNAL_COLOUR = "red"
LATENT_SPACE_COLOUR = "grey"
DEFAULT_FONT_SIZE = 12
DEFAULT_FONT_FAMILY = "sans-serif"
DEFAULT_FONT = "Avenir"

SIGNAL_LIM_UPPER = 300 / 3.086e+22
SIGNAL_LIM_LOWER = -600 / 3.086e+22

CM_TO_INCHES = 2.54

# Unified parameter mapping for LaTeX labels and ranges throughout the codebase
PARAMETER_LABELS = {
    # Intrinsic (CCSN) parameters
    'beta1_IC_b': r'$\beta_{IC,b}$',
    'omega_0(rad|s)': r'$\omega_0$',
    'A(km)': r'$A$',
    'Ye_c_b': r'$Y_e$',
    # Sky localization (extrinsic) parameters
    'ra': r'$\mathrm{RA}$',
    'dec': r'$\mathrm{Dec}$',
    'd': r'$d$',
    'psi': r'$\psi$',
}

# Default parameter ranges for plotting (in physical units after denormalization)
PARAMETER_RANGES = {
    # Intrinsic parameters
    'beta1_IC_b': (0, 0.25),
    'omega_0(rad|s)': (0, 16),
    'A(km)': (0, 10000),
    'Ye_c_b': (0.24, 0.29),
    # Sky parameters
    'ra': (0.0, 2.0 * np.pi),  # Radians, standard RA convention
    'dec': (-np.pi/2, np.pi/2),  # Radians
    'd': (0, 10),  # kiloparsecs (matches MAX_DISTANCE_KPC)
    'psi': (0, np.pi),  # Radians
}