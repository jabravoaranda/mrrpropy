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
MRR_PATH = Path(
    r"./tests/data/PRODUCTS/mrrpro81/2025/03/08/20250308_120000_processed.nc"
)
OUTPUT_DIR = Path(r"./tests/figures/mrr_plots_processed")
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
    variable = 'Dm'
    fig, ax = mrr.quickplot_reflectivity(field=variable)
    fig.savefig(
        OUTPUT_DIR / f"test_quickplot_reflectivity_{variable}.png"
    )  # Guardar la figura para inspección manual si se desea
    # Comprobación mínima de que devuelve objetos figura y ejes

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


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