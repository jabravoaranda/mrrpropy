import pytest
from datetime import datetime
import numpy as np
from pathlib import Path
import xarray as xr

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
    
def test_rain_process_analyze(mrr):
    analysis = mrr.rain_process_analyze(
        period=(datetime(2025, 3, 8, 12, 0, 0), datetime(2025, 3, 8, 12, 31, 0)),
        layer=(1000.0, 2000.0),
        k=11,
        ze_th=-5.0,
        min_points_ols=10,
        eps_q=0.01,
        rgb_q=0.02,
        vars_trend=("Dm", "Nw", "LWC"),
    )

    # --- tipo básico ---
    assert isinstance(analysis, xr.Dataset)
    assert "time" in analysis.coords
    assert analysis.sizes["time"] > 0

    # --- variables mínimas del análisis (contrato) ---
    for v in ("R", "G", "B", "minutes", "hex_x", "hex_y"):
        assert v in analysis, f"Falta '{v}' en la salida de rain_process_analyze."

    # --- trends mínimas (OLS) ---
    for v in ("b_Dm", "b_Nw", "b_LWC"):
        assert v in analysis, f"Falta '{v}' en la salida (trends)."

    # --- coherencia de minutes ---
    minutes = analysis["minutes"].values.astype(float)
    assert np.isfinite(minutes).any()
    # el primer instante debe ser 0 min (definición de minutes since start)
    assert np.nanmin(minutes) == pytest.approx(0.0)

    # --- attrs (decisiones ya tomadas en análisis) ---
    assert analysis.attrs.get("k", None) == 11
    assert analysis.attrs.get("z_top", None) == 1000.0
    assert analysis.attrs.get("z_base", None) == 2000.0
    assert analysis.attrs.get("eps_q", None) == 0.01
    assert analysis.attrs.get("rgb_q", None) == 0.02
    assert analysis.attrs.get("vars_trend", None) == ("Dm", "Nw", "LWC")

    # --- trazabilidad RGB (si la estás guardando como rgb_mapping) ---
    rgb_mapping = analysis.attrs.get("rgb_mapping", None)
    assert rgb_mapping is not None, "analysis.attrs['rgb_mapping'] debe existir."
    assert rgb_mapping == {"R": "Dm", "G": "Nw", "B": "LWC"}

    # --- coherencia temporal con el periodo pedido (tolerancia por slicing inclusivo) ---
    t_out0 = np.datetime64(analysis["time"].values[0])
    t_out1 = np.datetime64(analysis["time"].values[-1])
    assert t_out0 >= np.datetime64("2025-03-08T12:00:00") - np.timedelta64(1, "s")
    assert t_out1 <= np.datetime64("2025-03-08T12:31:00") + np.timedelta64(1, "s")
    
    classification = mrr.classify_rain_process(
        analysis=analysis )

    assert isinstance(classification, xr.Dataset)
    for v in ("proc_label", "sign_R", "sign_G", "sign_B", "strength"):
        assert v in classification

    