import pytest
import datetime
import numpy as np
from pathlib import Path
from matplotlib.collections import QuadMesh

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
matplotlib.use("Agg")  # imprescindible en CI/headless

from mrrpropy.raw_class import MRRProData  # cambia 'mrrpro' por el nombre real de tu módulo

# Ruta por defecto al fichero de prueba.
# Puedes sobrescribirla con la variable de entorno MRRPRO_TEST_FILE.
MRR_PATH = Path(r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000.nc")
OUTPUT_DIR = Path(r"./tests/figures/mrr_plots_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session")
def mrr():
    """Carga una instancia de MRRProData para todos los tests."""
    if not MRR_PATH.exists():
        pytest.skip(f"No se encuentra el archivo de data: {MRR_PATH}")
    mrr = MRRProData.from_file(MRR_PATH)
    yield mrr
    mrr.close()


def test_quickplot_reflectivity_runs(mrr):
    """quickplot_reflectivity debe ejecutarse sin errores.

    No verificamos el contenido de la figura, solo que no haya excepciones.
    Este test se puede marcar como 'slow' si lo deseas.
    """
    pytest.importorskip("matplotlib")

    variable = 'Ze'
    fig, ax = mrr.quicklook(variable=variable)
    fig.savefig(OUTPUT_DIR / f'test_quickplot_{variable}.png')  # Guardar la figura para inspección manual si se desea
    # Comprobación mínima de que devuelve objetos figura y ejes
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


# ============================================================
# NIVEL 2 — Guardado a disco: crea PNG y devuelve Path
# ============================================================
def test_plot_spectrum_saves_png(mrr):
    ds = mrr.ds
    spectrum_var = 'spectrum_raw'
    target_time = datetime.datetime(2025, 3, 8, 12, 50, 0)
    target_range = float(ds["range"].values[ds.sizes["range"] // 2])  # rango medio para asegurar datos

    fig, filepath = mrr.plot_spectrum(
        target_time,
        target_range,
        spectrum_var=spectrum_var,
        savefig=True,
        output_dir=OUTPUT_DIR,
        **{'dpi': 120}
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() == ".png"
    assert filepath.stat().st_size > 0
    plt.close(fig)


def test_plot_spectrogram_saves_png(mrr):
    target_time = datetime.datetime(2025, 3, 8, 12, 50, 0) 
    
    fig, filepath = mrr.plot_spectrogram(
        target_time,
        spectrum_var='spectrum_raw',
        savefig=True,
        output_dir=OUTPUT_DIR,
        **{"dpi": 120}
    )
    
    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() == ".png"
    assert filepath.stat().st_size > 0
    plt.close(fig)