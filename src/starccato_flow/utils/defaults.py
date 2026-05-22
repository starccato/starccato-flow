
import torch

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

SIGNALS_CSV = f"../data/training/richers_1764.csv"
PARAMETERS_CSV = f"../data/training/richers_1764_parameters.csv"
AVIRGO_ASD_FILE = f"../data/noise_asd/advirgo.txt"
ALIGO_ASD_FILE = f"../data/noise_asd/alIGO.txt"
TIME_CSV = f"../data/training/richers_1764_times.csv"


