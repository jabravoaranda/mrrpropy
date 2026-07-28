import numpy as np
import pytest


def test_dataset_loaded(raw_mrr):
    """El dataset se carga correctamente y tiene dimensiones no vacías."""
    assert raw_mrr.ds is not None
    assert raw_mrr.n_time > 0
    assert raw_mrr.n_range > 0
    assert "time" in raw_mrr.ds.dims
    assert "range" in raw_mrr.ds.dims


def test_basic_properties(raw_mrr):
    """Comprobación de propiedades básicas: time, range, variables."""
    t_index = raw_mrr.time
    assert len(t_index) == raw_mrr.n_time

    r = raw_mrr.range
    assert isinstance(r, np.ndarray)
    assert r.shape == (raw_mrr.n_range,)

    vars_ = raw_mrr.variables
    assert len(vars_) > 0
    for expected in ["Ze", "RR", "VEL", "N", "D", "index_spectra"]:
        assert expected in vars_


def test_get_field(raw_mrr):
    """get_field debe devolver un DataArray con las dimensiones esperadas."""
    ze = raw_mrr.get_field("Ze")
    assert ze.dims == ("time", "range")
    assert ze.shape == (raw_mrr.n_time, raw_mrr.n_range)

    with pytest.raises(KeyError):
        raw_mrr.get_field("variable_que_no_existe")


def test_subset_time_and_range(raw_mrr):
    """subset debe devolver un nuevo MRRProData con dimensiones reducidas."""
    mrr_sub = raw_mrr.subset(time_slice=slice(raw_mrr.time[0], raw_mrr.time[9]))
    assert mrr_sub.n_time == 10
    assert mrr_sub.n_range == raw_mrr.n_range

    mrr_sub2 = raw_mrr.subset(range_slice=slice(0, 2000))
    assert mrr_sub2.n_range == 39
    assert mrr_sub2.n_time == raw_mrr.n_time

    mrr_sub3 = raw_mrr.subset(
        time_slice=slice(raw_mrr.time[0], raw_mrr.time[9]),
        range_slice=slice(0, 2000),
    )
    assert mrr_sub3.n_time == 10
    assert mrr_sub3.n_range == 39


def test_nearest_time_index_and_profile(raw_mrr):
    """nearest_time_index y profile_at deben ser coherentes."""
    t0 = raw_mrr.time[0]
    idx0 = raw_mrr.nearest_time_index(t0)
    assert idx0 == 0

    profile = raw_mrr.profile_at(t0, field="Ze")
    assert profile.dims == ("range",)
    assert profile.shape == (raw_mrr.n_range,)

    t0_str = str(t0)
    idx0b = raw_mrr.nearest_time_index(t0_str)
    assert abs(idx0b - idx0) <= 1


def test_gate_spectrum(raw_mrr):
    """gate_spectrum debe devolver espectro y eje de velocidad coherentes."""
    vel, spec = raw_mrr.gate_spectrum(time_idx=0, range_idx=0)

    assert len(vel.shape) == 1
    assert len(spec.shape) == 1
    assert vel.shape == spec.shape

    n_samp = raw_mrr.ds.sizes["spectrum_n_samples"]
    assert vel.shape[0] == n_samp
    assert spec.shape[0] == n_samp

    vel_raw, spec_raw = raw_mrr.gate_spectrum(time_idx=0, range_idx=0, use_raw=True)
    assert vel_raw.shape == vel.shape
    assert spec_raw.shape == spec.shape
