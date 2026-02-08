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
OUTPUT_DIR = Path(r"./tests/figures/mrr_plots_microphysical_processes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def mrr():
    """Carga una instancia de MRRProData para todos los tests."""
    if not MRR_PATH.exists():
        pytest.skip(f"No se encuentra el archivo de data: {MRR_PATH}")
    mrr = MRRProData.from_file(MRR_PATH)
    yield mrr
    mrr.close()

def test_classify_microphysical_process_from_trends(mrr):
    """Test básico para classify_microphysical_process_from_trends."""
    fig, ax, saved = mrr.classify_microphysical_process_from_trends(
        z_top=1000.0,
        z_base=2000.0,
        variable_threshold="Ze",
        threshold_value=-5.0,
        eps_mode="hourly_quantile",
        q=0.1,
        time_dim="time",
        savefig=True,
        output_dir=OUTPUT_DIR,
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert saved is not None and Path(saved).exists()

def test_plot_microphysics_summary_multipanel(mrr):
    """Test básico para plot_microphysics_summary_multipanel."""
    fig, out = mrr.plot_microphysics_summary_multipanel(
        period = (datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 12, 59, 50)),
        layer = (1000.0, 2000.0),
        variable_threshold="Ze",
        threshold_value=-5.0,
        eps_mode="hourly_quantile",
        q=0.1,
        time_dim="time",
        savefig=True,
        output_dir=OUTPUT_DIR,
    )

    assert isinstance(fig, Figure)
    assert "microphysical_process" in out
