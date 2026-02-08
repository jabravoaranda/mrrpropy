import pytest
import datetime
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
before_path = Path(r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000.nc")
after_path = (
    Path(before_path.absolute().as_posix().replace("RAW", "PRODUCTS")).parent
    / f"{before_path.stem}_raprompro.nc"
)

OUTPUT_DIR = Path(r"./tests/figures/mrr_plots_raprompro_products")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def mrr():
    """Carga una instancia de MRRProData para todos los tests."""
    if not before_path.exists():
        pytest.skip(f"No se encuentra el archivo de data: {before_path}")
    mrr = MRRProData.from_file(before_path)
    mrr.load_raprompro(after_path)

    yield mrr
    mrr.close()


def test_quickplot_reflectivity_runs(mrr):
    """quickplot_reflectivity debe ejecutarse sin errores.

    No verificamos el contenido de la figura, solo que no haya excepciones.
    Este test se puede marcar como 'slow' si lo deseas.
    """
    pytest.importorskip("matplotlib")
    variable = "Ze"
    fig, ax = mrr.quicklook(variable=variable, source = 'raw', vmin=0, vmax=40)
    fig.savefig(
        OUTPUT_DIR / f"test_quicklook_{variable}_before.png"
    )  # Guardar la figura para inspección manual si se desea
    # Comprobación mínima de que devuelve objetos figura y ejes

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)

    fig, ax = mrr.quicklook(variable=variable, source = 'raprompro', vmin=0, vmax=40)
    fig.savefig(
        OUTPUT_DIR / f"test_quicklook_{variable}_after.png"
    )  # Guardar la figura para inspección manual si se desea
    # Comprobación mínima de que devuelve objetos figura y ejes

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_microphysical_properties_profiles_runs(mrr):
    """plot_microphysical_properties_profiles debe ejecutarse sin errores.

    No verificamos el contenido de la figura, solo que no haya excepciones.
    Este test se puede marcar como 'slow' si lo deseas.
    """
    pytest.importorskip("matplotlib")
    target_datetime = datetime.datetime(2025, 3, 8, 12, 50, 0)

    fig, axs, filepath = mrr.plot_microphysical_properties_profiles(
        target_datetime=target_datetime,
        savefig=True,
        output_dir=OUTPUT_DIR,
        **{"dpi": 120},
    )

    # Comprobación mínima de que devuelve objetos figura y ejes
    assert isinstance(fig, Figure)
    assert isinstance(axs, np.ndarray) and all(
        isinstance(ax, Axes) for ax in axs.flatten()
    )
    assert filepath is not None
    assert filepath.exists()
    plt.close(fig)


def test_plot_dealiased_spectrogram(mrr):
    """plot_DSD_gram debe ejecutarse sin errores."""
    pytest.importorskip("matplotlib")
    target_datetime = datetime.datetime(2025, 3, 8, 12, 50, 0)
    
    fig, filepath = mrr.plot_spectrogram(
        target_datetime=target_datetime,
        spectrum_var="spe_3D",
        savefig=True,
        output_dir=OUTPUT_DIR,
        **{"dpi": 120},
    )

    # Comprobación mínima de que devuelve objetos figura y ejes
    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    plt.close(fig)

def test_plot_dealiased_spectrogram(mrr):
    """plot_DSD_gram debe ejecutarse sin errores."""
    pytest.importorskip("matplotlib")
    target_datetime = datetime.datetime(2025, 3, 8, 12, 50, 0)
    
    fig, filepath = mrr.plot_DSDgram(
        target_datetime=target_datetime,        
        savefig=True,
        output_dir=OUTPUT_DIR,
        **{"dpi": 120},
    )

    # Comprobación mínima de que devuelve objetos figura y ejes
    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    plt.close(fig)

def test_plot_DSD_by_range(mrr):
    """plot_DSD_by_range debe ejecutarse sin errores."""
    target_datetime = datetime.datetime(2025, 3, 8, 12, 50, 0)
    target_heights = np.arange(500, 2500, 250)
    
    fig, filepath = mrr.plot_DSD_by_range(
        target_datetime=target_datetime,
        ranges=target_heights,
        savefig=True,
        output_dir=OUTPUT_DIR,
        **{"dpi": 120},
    )

    # Comprobación mínima de que devuelve objetos figura y ejes
    assert isinstance(fig, Figure)    
    assert filepath is not None
    assert filepath.exists()
    plt.close(fig)
    