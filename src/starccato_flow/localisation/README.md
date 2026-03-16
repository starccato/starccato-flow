# Multi-Channel Sky Localization Module

This module provides classes for multi-detector gravitational wave analysis and sky localization of galactic core-collapse supernovae (CCSN).

**Note:** The `CCSNDataMultiChannel` class only works with generated signals (e.g., from CVAE), not raw CCSN CSV files. Use `CCSNData` for loading original simulation data.

## Components

### 1. `DetectorNetwork` 
[detector_network.py](detector_network.py)

Handles gravitational wave detector network operations:
- **Antenna pattern calculations**: Computes F+ and Fx response functions for any sky position
- **Time delay calculations**: Determines signal arrival time differences between detectors
- **Network SNR**: Combines individual detector SNRs into network SNR
- **Detector information**: LIGO Hanford (H1), LIGO Livingston (L1), Virgo (V1)

**Key Methods:**
```python
network = DetectorNetwork()

# Antenna patterns
F_plus, F_cross = network.compute_antenna_patterns(ra, dec, detector_name='H1')

# Time delays between detectors
time_delays = network.compute_time_delays(ra, dec, reference_detector='H1')
# Returns: {'H1': 0.0, 'L1': 0.0105, 'V1': 0.0236} (seconds)

# Detector separation
separation = network.get_detector_separation('H1', 'L1')  # meters
```

### 2. `CCSNLocations`
[locations.py](locations.py)

Manages supernova locations and coordinate transformations:
- **Location generation**: Creates realistic galactic CCSN distributions using Faucher-Giguère & Kaspi (2006) model
- **Coordinate transformations**: Converts between galactic Cartesian and equatorial (RA, Dec) coordinates
- **Distance calculations**: Computes heliocentric distances
- **CSV loading**: Can load pre-computed supernova locations

**Key Methods:**
```python
locations = CCSNLocations()

# Generate locations
galactic_coords = locations.generate_locations(num_supernovae=1000, seed=42)

# Or load from file
locations = CCSNLocations(locations_file='exploded_supernovae.csv')

# Access coordinates
ra, dec = locations.ra, locations.dec  # radians
distances = locations.distances  # kpc
sky_params = locations.get_sky_params()  # [RA, Dec, distance]

# Coordinate conversions
ra, dec = locations.galactic_to_equatorial(x, y, z)
x, y, z = locations.equatorial_to_galactic(ra, dec, distance)
```

### 3. `CCSNDataMultiChannel`
[../data/ccsn_multi_channel.py](../data/ccsn_multi_channel.py)

Extended dataset class for multi-detector analysis with generated data only:
- **Multi-channel signals**: Projects waveforms to multiple detectors using antenna patterns
- **Sky parameters**: Includes RA, Dec, distance as conditioning parameters
- **Distance scaling**: Applies inverse-square law for different source distances
- **Detector-specific noise**: Adds independent noise to each detector channel
- **Generated data only**: Requires custom_data tuple, does not load from CSV

**Key Methods:**
```python
from starccato_flow.data import CCSNDataMultiChannel

# Generate signals first (e.g., from CVAE)
signals = ...  # Shape: (signal_length, num_samples)
parameters = ...  # Shape: (num_samples, 4) - [beta, omega, A, Ye]

# Option 1: Auto-generate sky parameters
dataset = CCSNDataMultiChannel(
    custom_data=(signals, parameters),  # Required
    detectors=['H1', 'L1', 'V1'],
    include_sky_params=True,
    noise=True,
    seed=42
)

# Option 2: Provide sky parameters explicitly
sky_params = ...  # Shape: (num_samples, 3) - [RA, Dec, distance]
dataset = CCSNDataMultiChannel(
    custom_data=(signals, parameters, sky_params),
    detectors=['H1', 'L1', 'V1'],
    include_sky_params=True
)

# Get sample:  with generated data:

```python
from starccato_flow.localisation import DetectorNetwork, CCSNLocations
from starccato_flow.data import CCSNDataMultiChannel

# 1. Generate signals (e.g., from trained CVAE)
from starccato_flow.nn.cvae import ConditionalVAE
cvae = ConditionalVAE(...)
cvae.load_state_dict(torch.load('cvae_weights.pt'))

# Generate 1000 signals
z = torch.randn(1000, z_dim)
params = torch.randn(1000, 4)  # Physical parameters
signals = cvae.decoder(z, params).detach().numpy()

# 2. Create multi-channel dataset
dataset = CCSNDataMultiChannel(
    custom_data=(signals, params.numpy()),  # Required
    detectors=['H1', 'L1', 'V1'],
    include_sky_params=True,
    noise=True,
    seed=42
)

# 3
# 2. Generate or load supernova locations
locations = CCSNLocations()
locations.generate_locations(num_supernovae=1000, seed=42)

# 3. Create multi-channel dataset
dataset = CCSNDataMultiChannel(
    detectors=['H1', 'L1', 'V1'],
    include_sky_params=True,
    noise=True,
    seed=42
)

# 4. Use for training
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for signals, params in loader:
    # signals: (batch, num_detectors, signal_length)
    # params: (batch, param_dim)  # includes RA, Dec, distance
    # Train your model...
    pass
```

## Parameter Dimensions

When `include_sky_params=True`:
- **Physical params**: beta, omega, A, Ye (4D)
- **Sky params**: RA, Dec, distance (3D)
- **Total**: 7D parameter space

The CVAE/Flow model conditions on all 7 parameters to enable:
1. Physical parameter inference (beta, omega, A, Ye)
2. Sky localization (RA, Dec)
3. Distance estimation

## Coordinate Systems

**Galactic Cartesian**:
- Origin at galactic center
- Sun at (0, 8.178, 0.0208) kpc
- x, y: galactic disk plane
- z: perpendicular to disk

**Equatorial**:
- RA: Right Ascension in radians [-π, π]
- Dec: Declination in radians [-π/2, π/2]
- Galactic center nominally at RA=0, Dec=0 in our convention

## Physics Implementation

**Antenna Patterns**:
- Computed from detector tensor D^{ab} = (e_x ⊗ e_x - e_y ⊗ e_y) / 2
- Accounts for detector orientation and Earth location
- Returns F+ and Fx for plus and cross polarizations

**Time Delays**:
- Δt = (r_det - r_ref) · n / c
- n: unit vector pointing from source to Earth
- Used for triangulation-based sky localization

**Distance Scaling**:
- Signals scaled by (10 kpc / distance) factor
- Antenna pattern further modulates signal per detector
- Combined effect: signal amplitude depends on both distance and sky location

## Future Extensions

1. **Time-dependent antenna patterns**: Account for Earth rotation using GPS time
2. **Polarization inference**: Include polarization angle as parameter
3. **Network configuration**: Support other detectors (KAGRA, LIGO India)
4. **Coherent vs. incoherent**: Implement both analysis methods
5. **Sky map generation**: Produce 2D probability maps on celestial sphere
