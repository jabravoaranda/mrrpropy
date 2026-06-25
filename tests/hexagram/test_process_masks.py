import matplotlib
import pytest

from mrrpropy.rain_process_info import PROCESS_SIGNATURES

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow]


def test_plot_hexagram_process(raw_mrr, product_dir):
    for process_ in PROCESS_SIGNATURES:
        fig, filepath = raw_mrr.plot.rain.process_mask_hexagram(
            process=process_,
            k=11,
            tol_center=0.15,
            savefig=True,
            output_dir=product_dir,
            crop_to_process=False,
        )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert filepath.exists()
