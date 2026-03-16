"""Detector network for multi-channel gravitational wave analysis."""

import numpy as np
from typing import Tuple, Dict, Optional


class DetectorNetwork:
    """Handles gravitational wave detector network for multi-channel analysis."""
    
    # Speed of light in m/s
    C = 299792458.0
    
    # Earth's radius in meters
    R_EARTH = 6371000.0
    
    # Detector information with geodetic coordinates
    # Coordinates from https://dcc.ligo.org/LIGO-T980044/public
    DETECTORS = {
        'H1': {  # LIGO Hanford
            'latitude': 46.4551,  # degrees North
            'longitude': -119.4077,  # degrees East
            'elevation': 142.554,  # meters
            'arm_azimuth': 125.9994,  # degrees (x-arm orientation from North)
            'arm_altitude': -6.195e-4,  # degrees (x-arm tilt)
        },
        'L1': {  # LIGO Livingston
            'latitude': 30.5629,
            'longitude': -90.7742,
            'elevation': -6.574,
            'arm_azimuth': 197.7165,
            'arm_altitude': -3.121e-4,
        },
        'V1': {  # Virgo
            'latitude': 43.6314,
            'longitude': 10.5045,
            'elevation': 51.884,
            'arm_azimuth': 70.5674,  # degrees
            'arm_altitude': 0.0,
        }
    }
    
    def __init__(self):
        """Initialize detector network with pre-computed detector tensors."""
        # Pre-compute detector tensors for each detector
        self._detector_tensors = {}
        for det_name in self.DETECTORS:
            self._detector_tensors[det_name] = self._compute_detector_tensor(det_name)
        
        # Pre-compute detector locations in geocentric Cartesian coordinates
        self._detector_locations = {}
        for det_name in self.DETECTORS:
            self._detector_locations[det_name] = self._compute_geocentric_location(det_name)
    
    def _compute_geocentric_location(self, detector_name: str) -> np.ndarray:
        """Convert geodetic coordinates to geocentric Cartesian coordinates.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            3D position vector in meters
        """
        det = self.DETECTORS[detector_name]
        lat = np.deg2rad(det['latitude'])
        lon = np.deg2rad(det['longitude'])
        
        # Simple spherical approximation (for precise calculations, use WGS84 ellipsoid)
        r = self.R_EARTH + det['elevation']
        
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        
        return np.array([x, y, z])
    
    def _compute_detector_tensor(self, detector_name: str) -> np.ndarray:
        """Compute the detector tensor D^{ab} for antenna pattern calculation.
        
        The detector tensor is: D^{ab} = (e_x^a e_x^b - e_y^a e_y^b) / 2
        where e_x and e_y are the unit vectors along the detector arms.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            3x3 detector tensor matrix
        """
        det = self.DETECTORS[detector_name]
        
        # Convert angles to radians
        lat = np.deg2rad(det['latitude'])
        lon = np.deg2rad(det['longitude'])
        azimuth = np.deg2rad(det['arm_azimuth'])
        
        # Compute local North, East, Zenith basis vectors in geocentric coordinates
        # East vector
        e_east = np.array([-np.sin(lon), np.cos(lon), 0])
        
        # North vector
        e_north = np.array([
            -np.sin(lat) * np.cos(lon),
            -np.sin(lat) * np.sin(lon),
            np.cos(lat)
        ])
        
        # Zenith vector (radial outward)
        e_zenith = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat)
        ])
        
        # X-arm direction (in local horizontal plane)
        # Azimuth is measured from North, clockwise
        e_x = np.cos(azimuth) * e_north + np.sin(azimuth) * e_east
        
        # Y-arm direction (perpendicular to x-arm, 90 degrees clockwise)
        e_y = np.cos(azimuth + np.pi/2) * e_north + np.sin(azimuth + np.pi/2) * e_east
        
        # Normalize
        e_x = e_x / np.linalg.norm(e_x)
        e_y = e_y / np.linalg.norm(e_y)
        
        # Compute detector tensor: D = (e_x ⊗ e_x - e_y ⊗ e_y) / 2
        D = (np.outer(e_x, e_x) - np.outer(e_y, e_y)) / 2
        
        return D
    
    def compute_antenna_patterns(self, ra: float, dec: float, detector_name: str, 
                                 gps_time: Optional[float] = None) -> Tuple[float, float]:
        """Compute F+ and Fx antenna patterns for given sky position.
        
        The antenna pattern functions relate the gravitational wave strain to detector response:
        h(t) = F+ h+(t) + Fx hx(t)
        
        Args:
            ra: Right Ascension in radians
            dec: Declination in radians
            detector_name: Name of the detector
            gps_time: Optional GPS time for Earth rotation correction (not implemented)
            
        Returns:
            Tuple of (F_plus, F_cross)
        """
        if detector_name not in self.DETECTORS:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        # Get pre-computed detector tensor
        D = self._detector_tensors[detector_name]
        
        # Wave propagation direction (from source to detector)
        # In equatorial coordinates:
        n = np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec)
        ])
        
        # Polarization tensors for + and x polarizations
        # These depend on the source orientation
        # For simplicity, use optimal orientation (face-on, circular polarization)
        
        # Construct orthonormal basis: (u, v, n)
        # u: perpendicular to n in the plane containing n and z-axis
        if abs(n[2]) < 0.9999:  # Not aligned with z-axis
            u = np.array([-n[1], n[0], 0])
            u = u / np.linalg.norm(u)
        else:  # Nearly aligned with z-axis, use x-axis
            u = np.array([1, 0, 0])
        
        # v: perpendicular to both n and u
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        
        # Polarization tensors (traceless, symmetric)
        # e+ = u ⊗ u - v ⊗ v
        e_plus = np.outer(u, u) - np.outer(v, v)
        
        # ex = u ⊗ v + v ⊗ u
        e_cross = np.outer(u, v) + np.outer(v, u)
        
        # Antenna patterns: F = D : e (tensor contraction)
        F_plus = np.sum(D * e_plus)
        F_cross = np.sum(D * e_cross)
        
        return F_plus, F_cross
    
    def compute_time_delays(self, ra: float, dec: float, 
                           reference_detector: str = 'H1',
                           gps_time: Optional[float] = None) -> Dict[str, float]:
        """Compute arrival time delays between detectors.
        
        Time delay: Δt_ij = (r_i - r_j) · n / c
        where r_i, r_j are detector positions and n is direction to source.
        
        Args:
            ra: Right Ascension in radians
            dec: Declination in radians
            reference_detector: Reference detector for time delays
            gps_time: Optional GPS time for Earth rotation (not implemented)
            
        Returns:
            Dictionary of time delays in seconds relative to reference detector
        """
        # Wave propagation direction (from source to detector)
        n = np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec)
        ])
        
        # Get reference detector location
        r_ref = self._detector_locations[reference_detector]
        
        # Compute time delays relative to reference
        time_delays = {}
        for det_name in self.DETECTORS:
            r_det = self._detector_locations[det_name]
            
            # Time delay: Δt = (r_det - r_ref) · n / c
            # Negative because n points from source to Earth
            dt = -np.dot(r_det - r_ref, n) / self.C
            
            time_delays[det_name] = dt
        
        return time_delays
    
    def compute_network_snr(self, snr_dict: Dict[str, float]) -> float:
        """Compute network SNR from individual detector SNRs.
        
        Network SNR: ρ_net = sqrt(Σ ρ_i^2)
        
        Args:
            snr_dict: Dictionary of SNR values for each detector
            
        Returns:
            Network SNR
        """
        return np.sqrt(sum(snr**2 for snr in snr_dict.values()))
    
    def get_detector_separation(self, det1: str, det2: str) -> float:
        """Get physical separation between two detectors in meters.
        
        Args:
            det1, det2: Detector names
            
        Returns:
            Separation distance in meters
        """
        r1 = self._detector_locations[det1]
        r2 = self._detector_locations[det2]
        return np.linalg.norm(r1 - r2)
    
    def __repr__(self) -> str:
        separations = []
        dets = list(self.DETECTORS.keys())
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                sep = self.get_detector_separation(dets[i], dets[j])
                separations.append(f"{dets[i]}-{dets[j]}: {sep/1000:.1f} km")
        
        return (f"DetectorNetwork({len(self.DETECTORS)} detectors)\n"
                f"  Detectors: {', '.join(self.DETECTORS.keys())}\n"
                f"  Separations:\n    " + "\n    ".join(separations))