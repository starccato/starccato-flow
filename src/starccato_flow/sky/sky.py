import numpy as np
import matplotlib.pyplot as plt
import math

class Sky:
    """Class representing sky positions in spherical coordinates."""
    def __init__(self, ra, dec):
        pass
    
    def pdf(self, r, A=1.96, r_0=17.2, theta_0=0.08, beta=0.13):
        return A * np.sin((np.pi * r) / r_0 + theta_0) * np.exp(-beta * r)
    
    def plot_pdf(self):
        # TODO: put this in plotting module
        r = np.linspace(0, 16.8, 1000)
        pdf_values = self.pdf(r)

        plt.plot(r, pdf_values)
        plt.title("Radial Distribution PDF for Galactic Supernovae")
        plt.xlabel("Distance from Galactic Center (kpc)")
        plt.ylabel("Probability Density")
        plt.grid()
        plt.show()

    def generate_galactic_supernovae(self, num_supernovae):
        """Generate galactic supernovae in the galactic disk using rejection sampling.
        Samples from the radial distribution PDF, then converts to galactocentric Cartesian coordinates."""
        A = 1.96
        r_0 = 17.2 
        theta_0 = 0.08
        beta = 0.13

        # Define 2D PDF: pdf(r) * r to account for area element in polar coordinates
        def pdf_2d(r):
            return self.pdf(r, A, r_0, theta_0, beta) * r
        
        r_test = np.linspace(0.01, 16.8, 1000)  # Start from 0.01 to avoid division issues
        pdf_max = np.max(np.abs(pdf_2d(r_test)))
   
        # Rejection sampling to generate radial distances from the 2D PDF
        r_samples = []
        while len(r_samples) < num_supernovae:
            # Propose samples uniformly in the range
            r_proposal = np.random.uniform(0.01, 16.8, num_supernovae * 2)
            # Generate uniform random values for acceptance test
            u = np.random.uniform(0, pdf_max, num_supernovae * 2)
            # Accept samples where u < pdf_2d(r)
            accepted = r_proposal[u < np.abs(pdf_2d(r_proposal))]
            r_samples.extend(accepted)
        
        r = np.array(r_samples[:num_supernovae])
        
        # Sample angles uniformly (azimuthally symmetric disk)
        theta = np.random.uniform(0, 2 * np.pi, num_supernovae)
        
        # Convert polar (r, theta) to Cartesian (x, y) for 2D galactic disk
        x = r * np.cos(theta)  # x in kpc
        y = r * np.sin(theta)  # y in kpc
        
        # Sample z heights independently from Gaussian (scale height ~100 pc = 0.1 kpc)
        z_heights = np.random.normal(loc=0, scale=0.1, size=num_supernovae)  # z in kpc
        
        # Stack into (num_supernovae, 3) array
        cartesian_coords = np.column_stack((x, y, z_heights))  # Shape: (num_supernovae, 3)
        
        return cartesian_coords 

    
    @property
    def theta(self):
        """Polar angle (colatitude) in radians."""
        return np.pi / 2 - self.dec
    
    @property
    def phi(self):
        """Azimuthal angle (longitude) in radians."""
        return self.ra