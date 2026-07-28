import numpy as np
from starccato_flow.supernovae.supernovae import Supernovae

def test_ccsn_plot_galactic_distribution_creates_output_files(tmp_path):
    ccsn = Supernovae(complex=True, limit=64)
    ccsn.generate_locations(num_supernovae=32, seed=7)

    fname_xy = tmp_path / "galactic_supernovae_xy.png"

    figures = ccsn.plot_galactic_distribution(
        fname_xy=str(fname_xy),
        background="white",
        font_family="sans-serif",
        font_name="Avenir",
        show=False,
    )

    assert len(figures) == 1
    assert np.array_equal(ccsn.galactic_coords.shape, (32, 3))
    assert fname_xy.exists()