import pandas as pd
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

from mrrpropy.raw_class import (
    MRRProData,
)  # cambia 'mrrpro' por el nombre real de tu módulo

# Ruta por defecto al fichero de prueba.
# Puedes sobrescribirla con la variable de entorno MRRPRO_TEST_FILE.
before_path = Path(r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000.nc")
after_path  = Path(before_path.absolute().as_posix().replace('RAW', 'PRODUCTS')).parent / f"{before_path.stem}_processed.nc"

OUTPUT_DIR = Path(r"./tests/figures/mrr_plots_raprompro_intercomparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def mrr_before():
    """Carga una instancia de MRRProData para todos los tests."""
    if not before_path.exists():
        pytest.skip(f"No se encuentra el archivo de data: {before_path}")
    mrr_before = MRRProData.from_file(before_path)
    yield mrr_before
    mrr_before.close()

@pytest.fixture(scope="session")
def mrr_after():
    """Carga una instancia de MRRProData para todos los tests."""
    if not after_path.exists():
        pytest.skip(f"No se encuentra el archivo de data: {after_path}")
    mrr_after = MRRProData.from_file(after_path)
    yield mrr_after
    mrr_after.close()

def test_quickplot_reflectivity_runs(mrr_after):
    """quickplot_reflectivity debe ejecutarse sin errores.

    No verificamos el contenido de la figura, solo que no haya excepciones.
    Este test se puede marcar como 'slow' si lo deseas.
    """
    pytest.importorskip("matplotlib")
    variable = 'Ze'
    fig, ax = mrr_before.quicklook_variable(variable=variable)
    fig.savefig(
        OUTPUT_DIR / f"test_quicklook_variable_{variable}_before.png"
    )  # Guardar la figura para inspección manual si se desea
    # Comprobación mínima de que devuelve objetos figura y ejes

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    fig, ax = mrr_after.quicklook_variable(variable=variable)
    fig.savefig(
        OUTPUT_DIR / f"test_quicklook_variable_{variable}_after.png"
    )  # Guardar la figura para inspección manual si se desea
    # Comprobación mínima de que devuelve objetos figura y ejes

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_raprompro_profiles_before_after(mrr_before, mrr_after):
    """Verifica que el procesamiento RARPOM produce un Dataset."""
    ds0 = mrr_before.ds
    ds1 = mrr_after.ds
    


def test_plot_raprompro_profiles(mrr):
    """plot_raprompro_profiles debe ejecutarse sin errores.

    No verificamos el contenido de la figura, solo que no haya excepciones.
    Este test se puede marcar como 'slow' si lo deseas.

    """

    fig, axes, filepath = mrr.plot_raprompro_profiles(
        target_datetime=datetime.datetime(2025, 3, 8, 12, 36, 0),
        savefig=True,
        output_dir=OUTPUT_DIR,
    )
    # Comprobación mínima de que devuelve objetos figura y ejes

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (5,)  # Debe haber un eje por variable solicitada
    assert filepath

def test_plot_raprompro_profiles_for_period(mrr):
    """plot_raprompro_profiles debe ejecutarse sin errores.

    No verificamos el contenido de la figura, solo que no haya excepciones.
    Este test se puede marcar como 'slow' si lo deseas.
    """
    period = (datetime.datetime(2025, 3, 8, 12, 58, 0), 
              datetime.datetime(2025, 3, 8, 12, 59, 50))
    times = mrr.ds.sel(time=slice(*period)).time

    #convert times to list of datetime
    times = [pd.to_datetime(t.values) for t in times]

    for time_ in times:
        mrr.plot_raprompro_profiles(
                target_datetime=time_,
                savefig=True,
                output_dir=OUTPUT_DIR,
            )
    assert True

# def test_plot_DSD_by_range_creates_expected_artists(mrr):
#     """
#     Test estándar (no visual) para plot_DSD_by_range:
#     - smoke test
#     - devuelve Figure y no guarda si savefig=False
#     - genera al menos una línea
#     - añade leyenda
#     """
#     if not hasattr(mrr, "plot_DSD_by_range"):
#         pytest.skip("MRRProData.plot_DSD_by_range() no existe todavía.")

#     ds = mrr.ds
#     if "time" not in ds or ds.sizes.get("time", 0) == 0:
#         pytest.skip("Dataset sin 'time'.")
#     if "range" not in ds or ds.sizes.get("range", 0) < 3:
#         pytest.skip("Dataset sin suficientes gates en 'range'.")

#     # tiempo representativo
#     t = ds["time"].values[ds.sizes["time"] // 2]
#     t = datetime.datetime(2025, 3, 8, 12, 50, 0)
#     # tres rangos repartidos (usar valores existentes para evitar fragilidad)
#     rvals = ds["range"].values.astype(float)
#     # r_list = [
#     #     float(rvals[len(rvals) // 6]),
#     #     float(rvals[len(rvals) // 2]),
#     #     float(rvals[5 * len(rvals) // 6]),
#     # ]
#     r_list = np.arange(500,4000, 200)

#     fig, filepath = mrr.plot_DSD_by_range(
#         t,
#         r_list,
#         use_log10=False,
#         savefig=True,
#         cmap='jet',
#         output_dir=OUTPUT_DIR,
#         **{"xlimits": (0,12), 'N_minimum_threshold': 1e-6},
#     )

#     assert isinstance(fig, Figure)
#     assert filepath
#     assert len(fig.axes) >= 1

#     ax = fig.axes[0]

#     # Debe haber al menos una línea trazada (idealmente 3, pero no lo exigimos)
#     assert len(ax.lines) >= 1, "No se trazaron curvas N(D)."

#     # Debe existir leyenda (si hay líneas con label)
#     legend = ax.get_legend()
#     assert legend is not None, "Se esperaba una leyenda con los rangos."

#     # Etiquetas mínimas
#     assert ax.get_xlabel() != ""
#     assert ax.get_ylabel() != ""

#     plt.close(fig)

# def test_plot_DSD_spectrogram_saves_png(mrr):
#     ds = mrr.ds
#     target_time = datetime.datetime(2025, 3, 8, 12, 50, 0)

#     fig, filepath = mrr.plot_spectrogram(
#         target_time,
#         spectrum_var='N',
#         variable_threshold = 'Ze',
#         threshold_value = 0,
#         use_log10=False,
#         savefig=True,
#         output_dir=OUTPUT_DIR,
#         **{"dpi": 120}
#     )

#     assert isinstance(fig, Figure)
#     assert filepath is not None
#     assert filepath.exists()
#     assert filepath.suffix.lower() == ".png"
#     assert filepath.stat().st_size > 0
#     plt.close(fig)
