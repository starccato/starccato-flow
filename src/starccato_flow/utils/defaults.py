
import torch
import os

# from .logger import logger

'''Default parameters for Starccato Flow'''

# model parameters

Y_LENGTH = 256
HIDDEN_DIM = 256
Z_DIM = 8

BATCH_SIZE = 32

SAMPLING_RATE = 1/4096

TEN_KPC = 3.086e+22

GPS_TIME = 1457654242.0

VALIDATION_SPLIT = 0.1

def get_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            print("CUDA device found")
            return torch.device("cuda")
        elif torch.mps.is_available():
            print("MPS device found")
            return torch.device("mps")
    except Exception as e:
        pass
    print("Using CPU device")
    return torch.device("cpu")

DEVICE = get_device()

# Construct absolute paths based on module location
# defaults.py is at: src/starccato_flow/utils/defaults.py
# We need to go up 3 levels to reach the starccato-flow root, then up 1 more to reach starccato/
_module_dir = os.path.dirname(os.path.abspath(__file__))
_starccato_flow_root = os.path.dirname(os.path.dirname(os.path.dirname(_module_dir)))
_data_root = os.path.join(_starccato_flow_root, "..", "data")

SIGNALS_CSV = os.path.join(_data_root, "training", "richers_1764.csv")
PARAMETERS_CSV = os.path.join(_data_root, "training", "richers_1764_parameters.csv")
AVIRGO_ASD_FILE = os.path.join(_data_root, "noise_asd", "advirgo.txt")
ALIGO_ASD_FILE = os.path.join(_data_root, "noise_asd", "aligo.txt")
TIME_CSV = os.path.join(_data_root, "training", "richers_1764_times.csv")


