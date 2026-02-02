import pytest
from datetime import datetime
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")  # imprescindible en CI/headless

from mrrpropy.raw_class import (
    MRRProData,
)  # cambia 'mrrpro' por el nombre real de tu módulo

# Ruta por defecto al fichero de prueba.
# Puedes sobrescribirla con la variable de entorno MRRPRO_TEST_FILE.
MRR_PATH = Path(
    r"./tests/data/PRODUCTS/mrrpro81/2025/03/08/20250308_120000_processed.nc"
)
OUTPUT_DIR = Path(r"./tests/figures/mrr_plots_rain_process")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def mrr():
    """Carga una instancia de MRRProData para todos los tests."""
    if not MRR_PATH.exists():
        pytest.skip(f"No se encuentra el archivo de data: {MRR_PATH}")
    mrr = MRRProData.from_file(MRR_PATH)
    yield mrr
    mrr.close()

def test_compute_layer_trend_ols(mrr):
    """Test básico para compute_layer_trend_ols."""
    result = mrr.compute_layer_trend_ols(
        z_top=1000.0,
        z_base=2000.0,
        variable_threshold="Ze",
        threshold_value=-5.0,
        eps_mode="hourly_quantile",
        q=0.1,
        time_dim="time",
    )

    assert "b_Dm" in result
    assert "b_LWC" in result
    assert "b_Nw" in result
    assert "a_Dm" in result
    assert "a_LWC" in result
    assert "a_Nw" in result
    assert "r2_Dm" in result
    assert "r2_LWC" in result
    assert "r2_Nw" in result
    assert "F_Dm" in result
    assert "F_LWC" in result
    assert "F_Nw" in result
    assert result["b_Dm"].shape == (mrr.ds.sizes["time"],)
    assert result["a_Dm"].shape == (mrr.ds.sizes["time"],)
    assert result["r2_Dm"].shape == (mrr.ds.sizes["time"],)
    assert result["F_Dm"].shape == (mrr.ds.sizes["time"],)
    assert result["b_LWC"].shape == (mrr.ds.sizes["time"],)
    assert result["a_LWC"].shape == (mrr.ds.sizes["time"],)
    assert result["r2_LWC"].shape == (mrr.ds.sizes["time"],)
    
def test_plot_hexagram_event_in_layer(mrr):
    """To test plot hexagram."""
    result = mrr.plot_hexagram_event_in_layer(
        period = (datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 13, 0, 0)),
        layer = (1000., 2000.),
        k = 1,
        eps_q = 0.01,
        rgb_q = 0.02,
        vars_trend = ("Dm", "Nw", "LWC"),
        nmin = 10,
        figsize = (10, 10),
        marker_size = 35.0,
        alpha = 0.9,
        savefig = False,
        output_dir = OUTPUT_DIR,
        dpi = 200,
    )

    assert True