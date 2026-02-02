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

def test_plot_rain_process_runs(mrr):
    """plot_rain_process debe ejecutarse sin errores.

    No verificamos el contenido de la figura, solo que no haya excepciones.
    Este test se puede marcar como 'slow' si lo deseas.
    """
    pytest.importorskip("matplotlib")

    fig, path = mrr.plot_rain_process_in_layer(
        target_datetime = (datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 13, 0, 0)),
        layer = (1000.,2000.),  # en metros
        x='Dm',
        y='Nw',
        z='LWC',
        savefig=True,
        **{'marker_size': 100, 
           'figsize': (12, 10),
           'cmap': 'seismic',
           'output_dir': OUTPUT_DIR}
        )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()