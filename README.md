# Starccato-Flow

Normalising Flow implementation for Starccato.

The aim is to train a neural-network conditional density estimator q(θ|s) to approximate the posterior distribution p(θ|s) of parameter values θ (theta) given detector strain data d. Note that d = s + n, where s is the signal, and n is the detector noise.

## Usage

### Instructions
- Create virtual environment
- Activate virtual environment
- Run `python3 -m pip install --upgrade build`
- Update `pyproject.toml` accordingly
- Run `python3 -m build` to generate distribution archives

### Running Starccato-Flow

## ??? 