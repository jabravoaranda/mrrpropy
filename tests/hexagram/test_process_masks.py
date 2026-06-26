import matplotlib
import pytest

from mrrpropy.hexagram import PROCESS_SIGNATURES, plot_process_to_hexagram

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow, pytest.mark.plot]


def test_plot_hexagram_process(artifact_dir):
    for process_ in PROCESS_SIGNATURES:
        fig, _, filepath = plot_process_to_hexagram(
            process=process_,
            k=11,
            tol_center=0.15,
            savefig=True,
            output_dir=artifact_dir,
            crop_to_process=False,
        )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert filepath.exists()


