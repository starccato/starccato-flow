import numpy as np

from starccato_flow.localisation import CCSN


def test_ccsn_plot_galactic_distribution_creates_output_files(tmp_path):
    ccsn = CCSN()
    ccsn.generate_locations(num_supernovae=32, seed=7)

    fname_3d = tmp_path / "galactic_supernovae_3d.png"
    fname_xy = tmp_path / "galactic_supernovae_xy.png"
    fname_xz = tmp_path / "galactic_supernovae_xz.png"

    figures = ccsn.plot_galactic_distribution(
        fname_3d=str(fname_3d),
        fname_xy=str(fname_xy),
        fname_xz=str(fname_xz),
        background="white",
        font_family="sans-serif",
        font_name="Avenir",
        show=False,
    )

    assert len(figures) == 3
    assert np.array_equal(ccsn.galactic_coords.shape, (32, 3))
    assert fname_3d.exists()
    assert fname_xy.exists()
    assert fname_xz.exists()