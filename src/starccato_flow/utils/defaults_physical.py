import numpy as np

from astropy.time import Time
t_j2000 = Time('J2000', scale='tt')  # exact TT-based epoch
print(t_j2000.gps)  # 630763161.8155832
GPS_TIME = t_j2000.gps
# GPS_TIME = 1457654242.0


SUN_LOCATION = np.array([0.0, 8.178, 0.0208]) # Sun is about 8.178 kpc from galactic center, and ~20.8 pc above the galactic plane

