import os

import requests
import torch
from tqdm import tqdm

# from .logger import logger

'''Default parameters for Starccato Flow'''

# model parameters

X_LENGTH = 256
HIDDEN_DIM = 512
Z_DIM = 8

BATCH_SIZE = 32

def get_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
    except Exception as e:
        pass
    return torch.device("cpu")

DEVICE = get_device()

# GENERATOR_WEIGHTS_FN = os.path.join(
#     os.path.dirname(__file__), "default_weights.pt"
# )

# WEIGHTS_URL = (
#     "https://github.com/starccato/data/raw/main/weights/generator_weights.pt"
# )

# def get_default_weights_path():
#     if not os.path.exists(GENERATOR_WEIGHTS_FN):
#         __download(
#             WEIGHTS_URL,
#             GENERATOR_WEIGHTS_FN,
#             msg="Downloading default weights...",
#         )
#     return GENERATOR_WEIGHTS_FN

# def __download(url: str, fname: str, msg: str = "Downloading") -> None:
#     response = requests.get(url, stream=True)
#     response.raise_for_status()
#     total_size_in_bytes = int(response.headers.get("content-length", 0))
#     block_size = 1024  # 1 Kibibyte
#     progress_bar = tqdm(
#         total=total_size_in_bytes,
#         unit="iB",
#         unit_scale=True,
#         desc=msg,
#         dynamic_ncols=True,
#     )
#     with open(fname, "wb") as file:
#         for data in response.iter_content(block_size):
#             progress_bar.update(len(data))
#             file.write(data)
#     progress_bar.close()
#     if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
#         raise Exception(
#             f"Download failed: expected {total_size_in_bytes} bytes, got {progress_bar.n} bytes"
#         )

# def _clear_cache():
#     if os.path.exists(GENERATOR_WEIGHTS_FN):
#         logger.info("Removing cached weights file.")
#         os.remove(GENERATOR_WEIGHTS_FN)


