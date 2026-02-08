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
MRR_PATH = Path(r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000.nc")
RPP_PATH = Path(
    r"./tests/data/PRODUCTS/mrrpro81/2025/03/08/20250308_120000_raprompro.nc"
)
OUTPUT_DIR = Path(r"./tests/figures/mrr_plots_hexagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def mrr():
    """Carga una instancia de MRRProData para todos los tests."""
    if not MRR_PATH.exists():
        pytest.skip(f"No se encuentra el archivo de data: {MRR_PATH}")
    mrr = MRRProData.from_file(MRR_PATH)
    mrr.load_raprompro(RPP_PATH)
    yield mrr
    mrr.close()
    
@pytest.fixture(scope="session")
def analysis(mrr):
    """Salida de rain_process_analyze reutilizable (hexagrama)."""
    return mrr.rain_process_analyze(
        period=(datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 12, 31, 0)),
        layer=(1000.0, 2000.0),
        k=11,
        ze_th=-5.0,
        min_points_ols=10,
        eps_q=0.01,
        rgb_q=0.02,
        vars_trend=("Dm", "Nw", "LWC"),
    )

@pytest.fixture(scope="session")
def classified(mrr, analysis):
    """Clasificación reutilizable."""
    return mrr.classify_rain_process(analysis=analysis)

def test_plot_rain_process_in_layer_2D(mrr):
    """Test básico: plot_rain_process_in_layer_2D debe ejecutarse y guardar figura."""
    fig, path = mrr.plot_rain_process_in_layer_2D(
        target_datetime=(datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 13, 0, 0)),
        layer=(1000.0, 2000.0),  # metros
        x="Dm",
        y="Nw",
        z="LWC",
        savefig=True,
        **{
            "marker_size": 100,
            "figsize": (12, 10),
            "cmap": "seismic",
            "output_dir": OUTPUT_DIR,
        },
    )    

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    assert path.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")

    # Limpieza en CI/headless
    plt.close(fig)


def test_plot_rain_process_in_layer_hexagram(mrr, analysis):
    """Test básico: crea figura + guarda fichero para el hexagrama en capa."""    
    result = mrr.plot_rain_process_in_layer_hexagram(
        analysis=analysis,
        savefig=True,
        output_dir=OUTPUT_DIR,
        dpi=200,
        **{"alpha_hexagram": 0.5, "cmap": "viridis"},
    )

    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2

    fig, filepath = result
    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")

    # --- comprobaciones básicas de contenido gráfico ---
    axes = fig.get_axes()
    assert len(axes) >= 1
    ax0 = axes[0]
    assert isinstance(ax0, Axes)

    # 1) El hexagrama base debería ser un imshow => AxesImage en ax.images
    assert len(ax0.images) >= 1, "Se esperaba al menos una imagen (imshow) con el hexagrama base."

    # 2) La trayectoria debería ser un scatter => PathCollection en ax.collections
    # (matplotlib crea colecciones para scatter)
    assert len(ax0.collections) >= 1, "Se esperaba al menos una colección (scatter) con la trayectoria."

    # Limpieza: cerrar figura para no acumular memoria en CI
    plt.close(fig)


def test_plot_microphysics_summary_multipanel(mrr, analysis, classified):
    """Test básico: plot_microphysics_summary_multipanel (plot-only) debe ejecutar y guardar figura."""
    
    fig, path = mrr.plot_microphysics_summary_multipanel(
        analysis=analysis,
        classified=classified,
        show_path_line=True,
        savefig=True,
        output_dir=OUTPUT_DIR,
        **{
            "figsize": (14, 10),
            "cmap": "viridis",
            "alpha_hexagram": 0.5,
            "markersize": 40.0,
            "line_width": 0.8,
            "dpi": 200,
        },
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()

    # Checks mínimos del contenido (sin validar ciencia)
    axes = fig.get_axes()
    assert len(axes) >= 4  # 4 paneles (puede haber ejes extra por colorbars)

    # Primer panel: hexagrama (imshow) debe existir
    assert len(axes[0].images) >= 1

    plt.close(fig)

