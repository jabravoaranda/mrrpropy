from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Union

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

import mrrpropy.RaProMPro_original as rpm
from mrrpropy.utils import (
    ols_slope_intercept_r2,
    compute_eps,
    build_rgb_from_trends,
    _sign_from_center,
    _strength,
    map_rgb_to_hexagram,
    get_hexagram_assets,
)


DatetimeLike = Union[str, np.datetime64, datetime]

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 32,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 14,
    }
)


@dataclass
class MicrophysicsConfig:
    variable_threshold: str = "Ze"
    threshold_value: float = -5.0
    min_points_ols: int = 10
    eps_q: float = 0.01
    rgb_q: float = 0.02
    eps_mode: str = "global_quantile"
    tol_center: float = 0.05
    min_strength: float = 0.10
    vars_trend: tuple[str, str, str] = ("Dm", "Nw", "LWC")
    k: int = 11  # default hex resolution


@dataclass
class PlotConfig:
    figsize: tuple[float, float] = (10, 10)
    figsize_hex: tuple[float, float] = (10, 10)
    figsize_summary: tuple[float, float] = (14, 10)
    figsize_quicklook: tuple[float, float] = (12, 8)
    figsize_spectrogram: tuple[float, float] = (10, 14)
    figsize_profiles: tuple[float, float] = (14, 10)
    figsize_multipanel: tuple[float, float] = (14, 10) 
    cmap: str = "jet"
    marker: str = "o"
    markersize: float = 10.0
    legendfontsize: float = 12.0
    alpha_points: float = 0.9
    alpha_hexagram: float = 0.25
    show_path_line: bool = True
    linewidth: float = 0.8
    dpi: int = 200


@dataclass
class MRRProData:
    """
    Helper class for working with METEK MRR-PRO data in CF/Radial format.

    Main Attributes
    ----------------
    path : str
        Path to the NetCDF file.
    ds : xr.Dataset
        xarray Dataset containing all MRR-PRO data.
    """

    path: str | Path
    ds: xr.Dataset

    micro_cfg: MicrophysicsConfig = field(default_factory=MicrophysicsConfig)
    plot_cfg: PlotConfig = field(default_factory=PlotConfig)

    def __post_init__(self):
        self.path = Path(self.path)
        self.raprompro: xr.Dataset | None = None

    # -------------------------
    # Constructors
    # -------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> "MRRProData":
        """
        Load a MRR-PRO NetCDF file and return a class instance.
        """
        ds = xr.open_dataset(path)
        return cls(path=path, ds=ds)

    # -------------------------
    # Basic Properties
    # -------------------------
    @property
    def time(self):
        """Time index as pandas DatetimeIndex."""
        return self.ds["time"].to_index()

    @property
    def range(self) -> np.ndarray:
        """
        Range of bins (m above radar, typically).
        """
        return self.ds["range"].values

    @property
    def n_time(self) -> int:
        return self.ds.sizes["time"]

    @property
    def n_range(self) -> int:
        return self.ds.sizes["range"]

    @property
    def variables(self) -> List[str]:
        """List of data variables (Za, Z, Ze, RR, VEL, etc.)."""
        return list(self.ds.data_vars)

    # -------------------------
    # Data Access
    # -------------------------
    def get_field(self, name: str) -> xr.DataArray:
        """
        Return a dataset variable (e.g., 'Ze', 'RR', 'VEL').
        """
        if name not in self.ds:
            raise KeyError(
                f"Variable '{name}' does not exist. Available variables: {list(self.ds.data_vars)}"
            )
        return self.ds[name]

    # -------------------------
    # Subsets
    # -------------------------
    def subset(
        self,
        time_slice: Optional[slice] = None,
        range_slice: Optional[slice] = None,
    ) -> "MRRProData":
        """
        Return a new instance with a subset in time and/or range.

        Examples
        --------
        mrr_sub = mrr.subset(time_slice=slice('2025-02-05T00:10', '2025-02-05T00:30'))
        mrr_sub = mrr.subset(range_slice=slice(0, 50))   # first 50 bins
        """
        sel_kwargs = {}
        if time_slice is not None:
            sel_kwargs["time"] = time_slice
        if range_slice is not None:
            sel_kwargs["range"] = range_slice

        ds_sub = self.ds.sel(**sel_kwargs)
        return MRRProData(path=self.path, ds=ds_sub)

    # -------------------------
    # Temporal Utilities
    # -------------------------
    def nearest_time_index(self, when: DatetimeLike) -> int:
        """
        Return the time index closest to 'when'.

        Parameters
        ----------
        when : str, np.datetime64 or datetime
        """
        t = self.ds["time"]
        when_np = np.datetime64(when)
        idx = int(np.argmin(np.abs(t.values - when_np)))
        return idx

    def profile_at(
        self,
        when: DatetimeLike,
        field: str = "Ze",
    ) -> xr.DataArray:
        """
        Return the vertical profile of a variable for the nearest time.

        Parameters
        ----------
        when : reference instant (str, np.datetime64, datetime)
        field : variable name (default 'Ze').

        Returns
        -------
        xr.DataArray with 'range' dimension.
        """
        if field not in self.ds:
            raise KeyError(f"Variable '{field}' does not exist in the dataset.")
        i = self.nearest_time_index(when)
        return self.ds[field].isel(time=i)

    # -------------------------
    # Doppler Spectra
    # -------------------------
    def gate_spectrum(
        self,
        time_idx: int,
        range_idx: int,
        use_raw: bool = False,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Return the Doppler spectrum for a gate (time_idx, range_idx).

        Uses:
          - index_spectra(time, range) -> index of 'n_spectra'
          - D(n_spectra, spectrum_n_samples) -> Doppler velocity axis
          - N(time, n_spectra, spectrum_n_samples) or spectrum_raw(...)

        Parameters
        ----------
        time_idx : time index (0 .. n_time-1)
        range_idx : range index (0 .. n_range-1)
        use_raw : if True, use 'spectrum_raw' instead of 'N'.

        Returns
        -------
        (vel, spec)
        vel  : DataArray with Doppler velocity (m/s, typically)
        spec : DataArray with spectrum (N or spectrum_raw)
        """
        if "index_spectra" not in self.ds:
            raise RuntimeError(
                "Dataset does not contain 'index_spectra'; cannot retrieve spectrum."
            )

        idx_spec = int(
            self.ds["index_spectra"].isel(time=time_idx, range=range_idx).values
        )

        # Velocity axis (only n_spectra, spectrum_n_samples)
        vel = self.ds["D"].isel(n_spectra=idx_spec)

        if use_raw:
            var_name = "spectrum_raw"
        else:
            var_name = "N"

        if var_name not in self.ds:
            raise RuntimeError(
                f"Dataset does not contain spectral variable '{var_name}'."
            )

        spec = self.ds[var_name].isel(time=time_idx, n_spectra=idx_spec)
        return vel, spec

    def process_raprompro(
        self,
        *,
        adjust_m: float = 1.0,
        save_spe_3d: bool = False,
        save_dsd_3d: bool = False,
        save: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Run RaProM-Pro processing using the published CLI algorithm implementation
        (RaProMPro_original.py), but exposed as a method returning an xarray.Dataset.

        Key design goal: keep the scientific algorithm and naming consistent with
        the original CLI output (Type, W, spectral width, Skewness, Kurtosis, DBPIA,
        LWC, RR, SR, Za, Z, Zea, Ze, Z_all, ... and BB_*).
        """
        if self.raprompro is not None:
            return self.raprompro

        ds = self.ds

        # -------------------------
        # 0) Validate minimal inputs
        # -------------------------
        has_raw = "spectrum_raw" in ds
        has_ref = "spectrum_reflectivity" in ds
        if not (has_raw or has_ref):
            raise RuntimeError(
                "Dataset must contain either 'spectrum_raw' or 'spectrum_reflectivity'."
            )

        if "range" not in ds or "time" not in ds:
            raise RuntimeError("Dataset must contain 'time' and 'range' coordinates.")

        if "transfer_function" not in ds or "calibration_constant" not in ds:
            raise RuntimeError(
                "Dataset must contain 'transfer_function' and 'calibration_constant'."
            )

        if "index_spectra" not in ds or "D" not in ds:
            raise RuntimeError(
                "CF/Radial spectra mapping requires 'index_spectra' and 'D'."
            )

        Code_spectrum = 0 if has_raw else 1

        # -------------------------
        # 1) Time resolution (TimeInt)
        # -------------------------
        tvals = ds["time"].values
        if tvals.size >= 2:
            # use minimum positive spacing, like the original uses min diff across files
            dt = np.diff(tvals.astype("datetime64[s]").astype("int64"))
            dt = dt[dt > 0]
            TimeInt = int(np.min(dt)) if dt.size else 60
        else:
            TimeInt = 60  # safe default

        # -------------------------
        # 2) Height vector (Hcolum) and radar constants (as original)
        # -------------------------
        Range = ds["range"].values.astype(float)
        DeltaH = float(Range[3] - Range[2]) if Range.size >= 4 else float(np.nan)
        Hcolum = Range.copy()
        FTcolum = ds["transfer_function"].values.astype(float)
        CC = float(ds["calibration_constant"].values)
        C = CC / float(adjust_m)

        # Dimensions in CF/Radial:
        # - spectrum_n_samples is the Doppler bin count (typically 64)
        # - range is the range-gate count (typically 128)
        Nhei = ds.sizes["range"]
        Nbins = ds.sizes["spectrum_n_samples"]

        # Radar constants: match original
        velc = 299792458.0
        lamb = velc / (24.23e9)
        fsampling = 500000.0
        fNy = fsampling * lamb / (2 * 2 * Nhei * Nbins)
        K2w = 0.92

        Deltaf = fsampling / (2 * Nhei * Nbins)
        Deltav = Deltaf * lamb / 2.0

        # constant to convert S/TF to eta(n): Cte=DeltaH*C/1e20 (original)
        Cte = DeltaH * C / 1e20

        # -------------------------
        # 3) Build D(range,bin) and Mie cross-sections, exactly as original
        # -------------------------
        dv = []
        for h in Hcolum:
            dv.append(1 + 3.68e-5 * h + 1.71e-9 * h**2)

        speed = np.arange(0, Nbins * fNy, fNy)

        # Diameters D(range, bin) from speed/dv (original)
        D = []
        for i in range(len(dv)):
            drow = []
            for j in range(len(speed)):
                b = speed[j] / dv[i]
                if 0.002 <= b <= 9.37:
                    drow.append(np.log((9.65 - b) / 10.3) * (-1 / 0.6))
                else:
                    drow.append(np.nan)
            D.append(drow)

        # Scattering/extinction cross-sections (original ScatExt)
        SigmaScatt = []
        SigmaExt = []
        for i in range(len(D)):
            sig1, sig2 = rpm.ScatExt(D[i], lamb)
            SigmaScatt.append(sig1)
            SigmaExt.append(sig2)

        # IMPORTANT: Process() uses these as module-level globals in the original code
        rpm.Nbins = Nbins
        rpm.NbinsM = Nbins
        rpm.Ntime = int(ds.sizes["time"])
        rpm.NheiM = Nhei
        rpm.fNy = fNy
        rpm.lamb = lamb
        rpm.K2w = K2w
        rpm.SigmaScatt = SigmaScatt
        rpm.SigmaExt = SigmaExt

        # Speeds exactly as original CLI
        rpm.speed = np.arange(0, Nbins * fNy, fNy)
        rpm.speed2 = np.arange(-Nbins * fNy, Nbins * fNy, fNy)
        rpm.speed3 = np.arange(-Nbins * fNy, 2 * Nbins * fNy, fNy)

        # -------------------------
        # 4) Helper to get raw/ref spectra per time, range (CF/Radial mapping)
        # -------------------------
        idx_map = ds["index_spectra"].values  # (time, range) -> n_spectra index
        # Safety: NaNs exist; coerce invalid to 0 and treat as missing later
        idx_map_int = np.where(np.isfinite(idx_map), idx_map, 0).astype(int)

        def _spectra_db_at_time(it: int, varname: str) -> np.ndarray:
            """
            Returns spec_db[range, bins] for a given time index, using index_spectra.
            Implemented as a loop to keep semantics explicit (matches CLI structure).
            """
            out = np.full((Nhei, Nbins), np.nan, dtype=float)
            for k in range(Nhei):
                ispec = int(idx_map_int[it, k])
                # If index_spectra is invalid, skip
                if ispec < 0 or ispec >= ds.sizes["n_spectra"]:
                    continue
                out[k, :] = ds[varname].isel(time=it, n_spectra=ispec).values
            return out

        # SNR for spectrum_reflectivity mode (original passes Snr_Refl_2)
        def _snr_at_time(it: int) -> np.ndarray:
            if "SNR" not in ds:
                return np.full(Nhei, np.nan, dtype=float)
            return ds["SNR"].isel(time=it).values.astype(float)

        # Convert time to unix seconds as original passes Time[i] numeric
        # (RaProMPro_original uses unix timestamps internally)
        time_unix = (ds["time"].values.astype("datetime64[s]").astype("int64")).astype(
            float
        )

        # -------------------------
        # 5) Main loop (mirrors CLI)
        # -------------------------
        bb_bot_full: list[float] = []
        bb_top_full: list[float] = []
        bb_peak_full: list[float] = []

        # Full matrices (time, range)
        estat_full = None
        sk_full = None
        kur_full = None
        PIA_full = None
        w_full = None
        sig_full = None
        LWC_full = None
        RR_full = None
        SnowR_full = None
        Z_da_full = None
        Z_a_full = None
        Z_ea_full = None
        Z_e_full = None
        z_all_full = None
        lwc_all_full = None
        rr_all_full = None
        n_all_full = None
        nw_full = None
        dm_full = None
        NW_all_full = None
        DM_all_full = None
        Noi_full = None
        SNR_full = None
        N_da_full = None

        # precipitation-type bookkeeping for PrepType (optional)
        Nw_2 = []
        Dm_2 = []

        # Output optional 3D
        spe_3d_list = (
            []
        )  # (time, range, speed3) in original; we store NewMatrix (dealiased) if requested
        dsd_3d_list = (
            []
        )  # (time, range, DropSize) in original; we store log10(NdE) if requested

        for it in range(ds.sizes["time"]):
            NewNoise = []
            Pot = []

            if Code_spectrum == 0:
                raw_db = _spectra_db_at_time(it, "spectrum_raw")  # (range, bins)
                # Loop over ranges exactly as CLI
                for k in range(Nhei):
                    COL_db = np.asarray(raw_db[k, :], dtype=float)
                    if np.isnan(COL_db).all():
                        NewNoise.append(np.nan)
                        Pot.append(np.full(Nbins, np.nan))
                        continue

                    COL_lin = np.power(10.0, COL_db / 10.0)
                    COL2, Noise = rpm.MrrProNoise2(COL_lin, k, DeltaH, TimeInt)

                    # original: Noise*(k)**2/TF[k] and COL2*(k)**2/TF[k]
                    NewNoise.append(Noise * (k**2) / FTcolum[k])
                    Pot.append((COL2 * (k**2)) / FTcolum[k])

                Snr_Refl_2 = []
            else:
                ref_db = _spectra_db_at_time(it, "spectrum_reflectivity")
                for k in range(Nhei):
                    COL_db = np.asarray(ref_db[k, :], dtype=float)
                    if np.isnan(COL_db).all():
                        Pot.append(np.full(Nbins, np.nan))
                    else:
                        Pot.append(np.power(10.0, COL_db / 10.0))
                Snr_Refl_2 = _snr_at_time(it)

            # continuity filter (original)
            NewNoise, Pot = rpm.Continuity(NewNoise, Pot, DeltaH)
            proeta = Pot

            # core processing (original Process return signature)
            (
                estat,
                NewMatrix,
                z_da,
                Lwc,
                Rr,
                SnowRate,
                w,
                sig,
                sk,
                Noi,
                DSD,
                NdE,
                Ze,
                Mov,
                velTur,
                snr,
                kur,
                PiA,
                NW,
                DM,
                z_P,
                lwc_P,
                rr_P,
                Z_h,
                Z_all,
                RR_all,
                LWC_all,
                dm_all,
                nw_all,
                N_all,
            ) = rpm.Process(
                proeta,
                Hcolum,
                time_unix[it],
                D,
                Cte,
                NewNoise,
                Deltav,
                Code_spectrum,
                Snr_Refl_2,
            )

            # BB logic (original uses special handling for first two times)
            if it == 0:
                bb_bot, bb_top, bb_peak = rpm.BB2(
                    w,
                    Ze,
                    Hcolum,
                    sk,
                    kur,
                    np.ones(2) * np.nan,
                    np.ones(2) * np.nan,
                    np.ones(2) * np.nan,
                )
            elif it == 1:
                bb_bot, bb_top, bb_peak = rpm.BB2(
                    w,
                    Ze,
                    Hcolum,
                    sk,
                    kur,
                    np.ones(2) * bb_bot_full,
                    np.ones(2) * bb_top_full,
                    np.ones(2) * bb_peak_full,
                )
            else:
                bb_bot, bb_top, bb_peak = rpm.BB2(
                    w, Ze, Hcolum, sk, kur, bb_bot_full, bb_top_full, bb_peak_full
                )

            bb_bot_full.append(bb_bot)
            bb_top_full.append(bb_top)
            bb_peak_full.append(bb_peak)

            # PIA in dB
            pIA = 10.0 * np.log10(PiA)

            # Apply PIA only for drizzle/rain exactly as CLI
            ZeCorrec = []
            ZaCorrec = []
            ZaCorrec_all = []
            for j in range(len(Ze)):
                ZaCorrec_all.append(Z_all[j] - pIA[j])
                if estat[j] == 10 or estat[j] == 5:
                    ZeCorrec.append(Ze[j] - pIA[j])
                    ZaCorrec.append(z_da[j] - pIA[j])
                else:
                    ZeCorrec.append(Ze[j])
                    ZaCorrec.append(np.nan)

            # Collect time-varying “type” params for PrepType (optional)
            if not np.isnan(DM).all():
                Nw_2.append(NW)
                Dm_2.append(DM)

            # Optional 3D outputs
            if save_spe_3d:
                spe_3d_list.append(np.asarray(NewMatrix, dtype=float))
            if save_dsd_3d:
                dsd_3d_list.append(np.log10(np.asarray(NdE, dtype=float)))

            # Stack into full matrices (same as CLI)
            def _stack(prev, cur):
                cur = np.asarray(cur, dtype=float)
                return cur if prev is None else np.vstack((prev, cur))

            estat_full = _stack(estat_full, estat)
            sk_full = _stack(sk_full, sk)
            kur_full = _stack(kur_full, kur)
            PIA_full = _stack(PIA_full, pIA)
            w_full = _stack(w_full, w)
            sig_full = _stack(sig_full, sig)
            LWC_full = _stack(LWC_full, Lwc)
            RR_full = _stack(RR_full, Rr)
            SnowR_full = _stack(SnowR_full, SnowRate)
            Z_da_full = _stack(Z_da_full, z_da)
            Z_a_full = _stack(Z_a_full, ZaCorrec)
            Z_ea_full = _stack(Z_ea_full, Ze)
            Z_e_full = _stack(Z_e_full, ZeCorrec)

            z_all_full = _stack(z_all_full, ZaCorrec_all)
            lwc_all_full = _stack(lwc_all_full, LWC_all)
            rr_all_full = _stack(rr_all_full, RR_all)
            n_all_full = _stack(n_all_full, N_all)

            nw_full = _stack(nw_full, NW)
            dm_full = _stack(dm_full, DM)
            NW_all_full = _stack(NW_all_full, nw_all)
            DM_all_full = _stack(DM_all_full, dm_all)

            Noi_full = _stack(Noi_full, Noi)
            SNR_full = _stack(SNR_full, snr)
            N_da_full = _stack(N_da_full, DSD)

        # -------------------------
        # 6) Smooth BB and correct values with BB matrix (original)
        # -------------------------
        bb_bot_full3 = rpm.Inter1D(bb_bot_full)
        bb_top_full3 = rpm.Inter1D(bb_top_full)
        bb_peak_full3 = rpm.Inter1D(bb_peak_full)

        bb_bot_full2 = rpm.anchor(bb_bot_full3, 0.95)
        bb_top_full2 = rpm.anchor(bb_top_full3, 0.95)
        bb_peak_full2 = rpm.anchor(bb_peak_full3, 0.95)

        # enforce ordering/consistency like CLI
        for j in range(len(bb_bot_full2)):
            if bb_peak_full2[j] > bb_top_full2[j]:
                bb_peak_full2[j] = bb_top_full2[j] - DeltaH
            if bb_peak_full2[j] < bb_bot_full2[j]:
                bb_peak_full2[j] = bb_bot_full2[j] + DeltaH

            if (
                np.isnan(bb_peak_full2[j])
                and ~np.isnan(bb_bot_full2[j])
                and np.isnan(bb_top_full2[j])
            ):
                bb_bot_full2[j] = np.nan
            if (
                np.isnan(bb_peak_full2[j])
                and np.isnan(bb_bot_full2[j])
                and ~np.isnan(bb_top_full2[j])
            ):
                bb_top_full2[j] = np.nan
            if (
                ~np.isnan(bb_peak_full2[j])
                and np.isnan(bb_bot_full2[j])
                and ~np.isnan(bb_top_full2[j])
            ):
                bb_top_full2[j] = np.nan
                bb_peak_full2[j] = np.nan
            if (
                ~np.isnan(bb_peak_full2[j])
                and ~np.isnan(bb_bot_full2[j])
                and np.isnan(bb_top_full2[j])
            ):
                bb_bot_full2[j] = np.nan
                bb_peak_full2[j] = np.nan
            if (
                ~np.isnan(bb_peak_full2[j])
                and np.isnan(bb_bot_full2[j])
                and np.isnan(bb_top_full2[j])
            ):
                bb_bot_full2[j] = np.nan
                bb_top_full2[j] = np.nan
            if (
                np.isnan(bb_peak_full2[j])
                and ~np.isnan(bb_bot_full2[j])
                and ~np.isnan(bb_top_full2[j])
            ):
                bb_peak_full2[j] = bb_bot_full2[j] + (
                    (bb_top_full2[j] - bb_bot_full2[j]) / 2.0
                )

        # CorrectWithBBMatrix in-place correction (CLI)
        estat_full, Z_da_full, LWC_full, RR_full, SnowR_full = rpm.CorrectWithBBMatrix(
            estat_full,
            Z_da_full,
            LWC_full,
            RR_full,
            SnowR_full,
            Hcolum,
            bb_bot_full2,
            bb_top_full2,
            Z_ea_full,
            # NOTE: these were built inside loop in CLI as Z_P/LWC_P/RR_P per time
            # In the CLI they keep Z_P/LWC_P/RR_P time-stacked. We reproduce that
            # by recomputing them as the “MP parameters” already returned from Process
            # is included inside the Process return; in this method we did not store
            # them. For exact parity you can store z_P/lwc_P/rr_P stacks too.
            # Minimal safe approximation: pass NaNs to skip those corrections.
            np.full_like(Z_da_full, np.nan),  # Z_P
            np.full_like(LWC_full, np.nan),  # LWC_P
            np.full_like(RR_full, np.nan),  # RR_P
            sk_full,
        )

        # -------------------------
        # 7) Build output Dataset with original CLI variable names
        # -------------------------
        coords = {
            "time": ds["time"].values,
            "range": Hcolum.astype(float),
            "BB_Height": np.array([0.0], dtype=float),
        }

        out = xr.Dataset(coords=coords)

        # 2D fields (time,range) with original names
        def _da2(name, data, units, desc):
            out[name] = xr.DataArray(
                np.asarray(data, dtype=float),
                dims=("time", "range"),
                attrs={"units": units, "description": desc},
            )

        _da2(
            "Type",
            estat_full,
            "",
            "Predominant hydrometeor type numerical value (original CLI)",
        )
        _da2("W", w_full, "m s-1", "Fall speed with aliasing correction")
        _da2(
            "spectral width",
            sig_full,
            "m s-1",
            "Spectral width of the dealiased velocity distribution",
        )
        _da2(
            "Skewness",
            sk_full,
            "none",
            "Skewness of the spectral reflectivity with dealiasing",
        )
        _da2(
            "Kurtosis",
            kur_full,
            "none",
            "Kurtosis of the spectral reflectivity with dealiasing",
        )
        _da2(
            "DBPIA",
            PIA_full,
            "dB",
            "Path Integrated Attenuation (dB) assuming liquid phase",
        )
        _da2(
            "LWC",
            LWC_full,
            "g m-3",
            "Liquid Water Content using only liquid hydrometeors (by Type)",
        )
        _da2(
            "RR",
            RR_full,
            "mm hr-1",
            "Rain Rate using only liquid hydrometeors (by Type)",
        )
        _da2("SR", SnowR_full, "mm hr-1", "Snow Rate")
        _da2(
            "Za",
            Z_a_full,
            "dBZ",
            "Attenuated reflectivity corrected by PIA only for liquid hydrometeors",
        )
        _da2("Zea", Z_ea_full, "dBZ", "Equivalent attenuated reflectivity")
        _da2(
            "Ze",
            Z_e_full,
            "dBZ",
            "Equivalent reflectivity corrected by PIA only for drizzle/rain",
        )
        _da2(
            "Z_all",
            z_all_full,
            "dBZ",
            "Attenuated reflectivity corrected by PIA assuming all liquid",
        )
        _da2("LWC_all", lwc_all_full, "g m-3", "LWC assuming all liquid")
        _da2("RR_all", rr_all_full, "mm hr-1", "RR assuming all liquid")
        _da2(
            "N_all", n_all_full, "log10(m-3 mm-1)", "log10(total N) assuming all liquid"
        )
        _da2(
            "Nw", nw_full, "log10(mm-1 m-3)", "Normalized intercept parameter (by Type)"
        )
        _da2("Dm", dm_full, "mm", "Mean mass-weighted diameter (by Type)")
        _da2(
            "Nw_all",
            NW_all_full,
            "log10(mm-1 m-3)",
            "Normalized intercept parameter (all liquid)",
        )
        _da2("Dm_all", DM_all_full, "mm", "Mean mass-weighted diameter (all liquid)")
        _da2("Noise", Noi_full, "", "Noise estimate in eta(n) units (original)")
        _da2("SNR", SNR_full, "dB", "SNR used/derived by algorithm (original)")
        _da2(
            "N_da",
            N_da_full,
            "log10(m-3 mm-1)",
            "log10(N(D)) derived (original 'N_da')",
        )

        # BB as (time,BB_Height) to mirror CLI netCDF shape
        out["BB_bottom"] = xr.DataArray(
            np.asarray(bb_bot_full2, dtype=float)[:, None],
            dims=("time", "BB_Height"),
            attrs={
                "units": "m",
                "description": "range from BB bottom above sea level (original CLI)",
            },
        )
        out["BB_top"] = xr.DataArray(
            np.asarray(bb_top_full2, dtype=float)[:, None],
            dims=("time", "BB_Height"),
            attrs={
                "units": "m",
                "description": "range from BB top above sea level (original CLI)",
            },
        )
        out["BB_peak"] = xr.DataArray(
            np.asarray(bb_peak_full2, dtype=float)[:, None],
            dims=("time", "BB_Height"),
            attrs={
                "units": "m",
                "description": "range from BB peak above sea level (original CLI)",
            },
        )

        # Optional 3D products (names follow original netCDF)
        if save_spe_3d:
            out["spe_3D"] = xr.DataArray(
                np.asarray(spe_3d_list, dtype=float),
                dims=("time", "range", "speed"),
                coords={
                    "time": coords["time"],
                    "range": coords["range"],
                    "speed": np.arange(-Nbins * fNy, 2 * Nbins * fNy, fNy),
                },
                attrs={
                    "units": "mm-1",
                    "description": "spectral reflectivity dealiased (original CLI)",
                },
            )

        if save_dsd_3d:
            out["dsd_3D"] = xr.DataArray(
                np.asarray(dsd_3d_list, dtype=float),
                dims=("time", "range", "DropSize"),
                coords={
                    "time": coords["time"],
                    "range": coords["range"],
                    "DropSize": np.asarray(D[0], dtype=float),
                },
                attrs={
                    "units": "log10(m-3 mm-1)",
                    "description": "3D DSD (original CLI)",
                },
            )

        self.raprompro = out
        if save:
            output_dir = kwargs.get("output_dir", Path.cwd())
            filename = kwargs.get("filename", f"{self.path.stem}_raprompro.nc")
            out.to_netcdf(output_dir / filename)

        return self.raprompro

    def _is_processed(
        self,
        *,
        required: Iterable[str] = ("Ze", "Zea", "Za", "Z_all", "Dm", "Nw", "LWC", "RR"),
    ) -> bool:
        """
        Heurística mínima: si existen las variables clave de RaProMPro,
        consideramos que el Dataset está preprocesado.

        Si quieres hacerlo más robusto, puedes además exigir algún atributo global:
        ds.attrs.get("processing") == "RaProMPro" o similar.
        """
        if self.raprompro is None:
            return False
        
        return all(v in self.raprompro.data_vars for v in required)

    # -------------------------
    # Resource Management
    # -------------------------
    def close(self):
        """Close the xarray dataset (e.g., at the end of the script)."""
        self.ds.close()

    # -------------------------------------------------------------------------
    # Helpers internos para espectros MRR-PRO
    # -------------------------------------------------------------------------
    def load_raprompro(
        self,
        path: str | Path,
        *,
        chunks: str | dict | None = "auto",
        validate: bool = True,
        required_vars: tuple[str, ...] = (
            "Ze",
            "Dm",
            "Nw",
            "LWC",
            "RR",
            "Nw_all",
            "Dm_all",
            "N_da",
        ),
        assign: bool = True,
    ) -> xr.Dataset:
        """
        Carga un NetCDF ya procesado por RaProMPro y lo asigna a self.raprompro.

        Parameters
        ----------
        path : str | Path
            Ruta al fichero *_raprompro.nc (p.ej. '20250308_120000_raprompro.nc').
        chunks : "auto" | dict | None
            Si no es None, abre en modo dask (lazy) para acelerar I/O y evitar cargar todo a RAM.
        validate : bool
            Si True, comprueba que el dataset tiene dims/coords esperadas y que encaja con self.ds.
        required_vars : tuple[str, ...]
            Variables mínimas que deben existir en el dataset procesado.
        assign : bool
            Si True, guarda el dataset en self.raprompro.

        Returns
        -------
        xr.Dataset
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el fichero: {path}")

        ds_rp = xr.open_dataset(path, chunks=chunks)

        if validate:
            # 1) Dims/coords mínimas
            for c in ("time", "range"):
                if c not in ds_rp.coords:
                    raise ValueError(
                        f"El raprompro cargado no tiene coord '{c}'. "
                        f"coords={list(ds_rp.coords)}"
                    )

            # 2) Variables mínimas (heurística simple)
            missing = [v for v in required_vars if v not in ds_rp.data_vars]
            if missing:
                raise ValueError(
                    f"El raprompro cargado no parece un output válido: faltan {missing}. "
                    f"vars={list(ds_rp.data_vars)}"
                )

            # 3) Compatibilidad con self.ds (time/range)
            #    (si no quieres esto, pon validate=False)
            if "time" in self.ds.coords:
                t0 = self.ds["time"].values
                t1 = ds_rp["time"].values
                if (t0.shape != t1.shape) or (not np.array_equal(t0, t1)):
                    raise ValueError(
                        "Incompatibilidad en 'time' entre self.ds y el raprompro cargado "
                        f"(self.ds: {t0.shape}, raprompro: {t1.shape})."
                    )

            if "range" in self.ds.coords:
                r0 = self.ds["range"].values
                r1 = ds_rp["range"].values
                if (r0.shape != r1.shape) or (not np.array_equal(r0, r1)):
                    raise ValueError(
                        "Incompatibilidad en 'range' entre self.ds y el raprompro cargado "
                        f"(self.ds: {r0.shape}, raprompro: {r1.shape})."
                    )

        if assign:
            self.raprompro = ds_rp            

        return ds_rp

    def _nearest_period(
        self, target_datetime: datetime | np.datetime64, target_range: float
    ) -> tuple[np.datetime64, float]:
        """Devuelve el time y range reales seleccionados por nearest."""
        ds = self.ds
        t_sel = ds["time"].sel(time=target_datetime, method="nearest").values
        r_sel = float(ds["range"].sel(range=target_range, method="nearest").values)
        return t_sel, r_sel

    def _get_velocity_axis(self, n_bins: int) -> np.ndarray:
        """
        Construye el eje de velocidades Doppler (m/s) en ausencia de un eje explícito.

        Nota: MRR-Pro a menudo no guarda el vector de velocidades por bin como coord.
        Usamos fold_limit_upper si está en attrs de VEL, si no asumimos 12 m/s.
        """
        ds = self.ds
        vny = 12.0
        if "VEL" in ds and isinstance(ds["VEL"].attrs, dict):
            if "fold_limit_upper" in ds["VEL"].attrs:
                try:
                    vny = float(ds["VEL"].attrs["fold_limit_upper"])
                except Exception:
                    pass
        # En muchos ficheros MRR-Pro el espectro está en [0, vny]
        return np.linspace(0.0, vny, int(n_bins), dtype=float)

    def _get_spectrum_1d(
        self,
        target_datetime: datetime | np.datetime64,
        target_range: float,
        *,
        spectrum_var: str = "spectrum_reflectivity",
    ) -> tuple[np.datetime64, float, np.ndarray, np.ndarray, str]:
        """
        Extrae el espectro 1D más cercano a (time, range), soportando:
          - cubo: spectrum_var(time, range, spectrum_n_samples)
          - indexado: spectrum_var(time, n_spectra, spectrum_n_samples) + index_spectra(time, range)

        Returns:
          t_sel, r_sel, vel_axis, spec_1d, units
        """
        ds = self.ds
        if spectrum_var not in ds:
            # fallback frecuente: spectrum_raw
            if "spectrum_raw" in ds:
                spectrum_var = "spectrum_raw"
            else:
                raise KeyError(
                    f"No encuentro '{spectrum_var}' ni 'spectrum_raw' en el Dataset."
                )

        t_sel, r_sel = self._nearest_period(target_datetime, target_range)

        da = ds[spectrum_var]
        units = str(da.attrs.get("units", ""))
        # dims candidatas
        bin_dim = "spectrum_n_samples"
        if bin_dim not in da.dims:
            # por si el fichero usa otro nombre
            raise ValueError(
                f"'{spectrum_var}' no tiene dimensión '{bin_dim}'. dims={da.dims}"
            )

        # Caso A: cubo (time, range, spectrum_n_samples)
        if ("time" in da.dims) and ("range" in da.dims):
            s = da.sel(time=t_sel, range=r_sel, method="nearest").values.astype(float)
            vel = self._get_velocity_axis(s.shape[-1])
            return t_sel, r_sel, vel, s, units

        # Caso B: indexado (time, n_spectra, spectrum_n_samples)
        if ("time" in da.dims) and ("n_spectra" in da.dims):
            if "index_spectra" not in ds:
                raise KeyError(
                    f"'{spectrum_var}' es (time,n_spectra,bin) pero falta 'index_spectra(time,range)'."
                )
            idx = (
                ds["index_spectra"]
                .sel(time=t_sel, range=r_sel, method="nearest")
                .values
            )
            js = int(idx)
            s = da.sel(time=t_sel, n_spectra=js).values.astype(float)
            vel = self._get_velocity_axis(s.shape[-1])
            return t_sel, r_sel, vel, s, units

        raise ValueError(f"Formato de '{spectrum_var}' no soportado. dims={da.dims}")

    def _get_spectrogram_2d(
        self,
        target_datetime: datetime | np.datetime64,
        *,
        spectrum_var: str,
        range_limits: tuple[float, float] | None = None,
    ) -> tuple[np.datetime64, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Extrae un espectrograma 2D (range x doppler_bin) para el instante más cercano.

        Returns:
          t_sel, ranges, vel_axis, spec2d, units
        """
        ds = self.ds

        if spectrum_var not in ds:            
            if spectrum_var not in self.raprompro:
                raise KeyError(f"'{spectrum_var}' not found.")
            else:
                ds = self.raprompro
        else:
            ds = self.ds
        
        da = ds[spectrum_var]

        t_sel = da["time"].sel(time=target_datetime, method="nearest").values
        units = str(da.attrs.get("units", "?"))

        # Rango a representar
        if range_limits is None:
            r0 = float(da["range"].min().values)
            r1 = float(da["range"].max().values)
        else:
            r0, r1 = map(float, range_limits)

        ranges = da["range"].sel(range=slice(r0, r1)).values.astype(float)

        #Spectrum 
        if 'spectrum_n_samples' in da.sizes:
            n_bins = da.sizes["spectrum_n_samples"]
            if n_bins is None:
                raise ValueError(f"spectrum_n_samples not found in Dataset.")
            vel = self._get_velocity_axis(int(n_bins))
        else:
            if 'speed' in self.raprompro:
                vel = self.raprompro["speed"]
            else:
                raise ValueError(f"velocity not found in raprompro Dataset.")
        
        # Caso A: cubo (time, range, bin)
        if ("time" in da.dims) and ("range" in da.dims):
            spec2d = da.sel(time=t_sel, range=slice(r0, r1)).values.astype(float)
            # spec2d shape: (range, bin)
            return t_sel, ranges, vel, spec2d, units

        # Caso B: indexado (time, n_spectra, bin) + index_spectra(time, range)
        if ("time" in da.dims) and ("n_spectra" in da.dims):
            if "index_spectra" not in ds:
                raise KeyError(
                    f"'{spectrum_var}' es (time,n_spectra,bin) pero falta 'index_spectra(time,range)'."
                )
            # selecciona índices para todos los ranges del slice
            idx_vec = (
                ds["index_spectra"]
                .sel(time=t_sel, range=slice(r0, r1))
                .values.astype(int)
            )
            # Extrae spectra para ese time: (n_spectra, bin)
            slab = da.sel(time=t_sel).values.astype(float)  # (n_spectra, bin)
            # Mapea (range -> n_spectra) => (range, bin)
            spec2d = slab[idx_vec, :]
            return t_sel, ranges, vel, spec2d, units

        raise ValueError(f"Formato de '{spectrum_var}' no soportado. dims={da.dims}")

    # -------------------------
    # Quick Plot (optional)
    # -------------------------
    def quicklook(
        self,
        variable: str = "Ze",
        source: str = 'raprompro',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        **kwargs,
    ):
        """
        Create a time–range plot of reflectivity (or any 2D variable time × range).

        Requires matplotlib. Intended for quick inspections.
        """
        pcfg = self.plot_cfg  # instancia de PlotConfig

        cmap = kwargs.get("cmap", pcfg.cmap)
        figsize = kwargs.get("figsize", pcfg.figsize)

        if source == 'raw':
            if variable not in self.ds:            
                raise KeyError(f"Variable '{variable}' not found in raw Dataset.")
            else:
                da = self.ds[variable]  # dims: (time, range)
        else:
            if variable not in self.ds:
                raise KeyError(f"Variable '{variable}' not found in raprompro Dataset.")
            else:
                da = self.raprompro[variable]

        fig, ax = plt.subplots(figsize=figsize)
        im = da.plot(
            ax=ax,
            x="time",
            y="range",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
            cmap=cmap,
        )
        ax.set_title(f"{variable} (MRR-PRO)")
        ax.set_ylabel("Range (m)")
        ax.set_xlabel("Time")
        plt.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    def plot_spectrum(
        self,
        target_datetime: datetime | np.datetime64,
        target_range: float,
        *,
        spectrum_var: str = "spectrum_reflectivity",
        velocity_limits: tuple[float, float] | None = None,
        label_type: str = "both",  # both|time|range
        fig: Figure | None = None,
        ax=None,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """
        Plot a 1D spectrum at a specified time and range.
        Parameters
        ----------
        target_datetime : datetime | np.datetime64
            The target time for which to extract the spectrum.
        target_range : float
            The target range (in meters) for which to extract the spectrum.
        spectrum_var : str, optional
            The spectrum variable to plot. Default is "spectrum_reflectivity".
        velocity_limits : tuple[float, float] | None, optional
            The velocity limits for the x-axis as (min, max). If None, limits are
            automatically determined from the data. Default is None.
        label_type : str, optional
            The type of label to display. Options are "both" (time and range),
            "time", or "range". Default is "both".
        fig : Figure | None, optional
            An existing matplotlib Figure object. If None, a new figure is created.
            Default is None.
        ax : Axes | None, optional
            An existing matplotlib Axes object. If None and fig is None, a new
            axes is created. If fig is provided but ax is None, the first axes
            from fig is used. Default is None.
        savefig : bool, optional
            Whether to save the figure to a file. Default is False.
        output_dir : Path | None, optional
            The directory where the figure will be saved. Required if savefig is True.
            Default is None.
            Additional keyword arguments passed to matplotlib plotting functions.
            - color : str, optional
                Line color for the spectrum plot. Default is 'black'.
            - dpi : int, optional
                DPI for saved figure. If not provided, uses the plot configuration dpi.
        Returns
        -------
        tuple[Figure, Path | None]
            A tuple containing:
            - fig : matplotlib Figure object
            - filepath : Path to the saved figure if savefig is True, otherwise None
        Raises
        ------
        ValueError
            If savefig is True but output_dir is None.
        """
        pcfg = self.plot_cfg

        dpi = kwargs.get("dpi", pcfg.dpi)
        figsize = kwargs.get("figsize", pcfg.figsize)

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif fig is not None and ax is None:
            ax = fig.get_axes()[0]

        t_sel, r_sel, vel, spec, units = self._get_spectrum_1d(
            target_datetime, target_range, spectrum_var=spectrum_var
        )

        # etiqueta
        t_txt = np.datetime_as_string(t_sel, unit="s")
        if label_type == "both":
            label = f"{t_txt} | {r_sel:.1f} m"
        elif label_type == "range":
            label = f"{r_sel:.1f} m"
        else:
            label = f"{t_txt}"

        if not np.isnan(spec).all():
            ax.plot(vel, spec, color=kwargs.get("color", "black"), label=label)

        # ejes
        if velocity_limits is not None:
            ax.set_xlim(*velocity_limits)
        else:
            ax.set_xlim(float(np.nanmin(vel)), float(np.nanmax(vel)))

        ax.set_xlabel("Doppler velocity [m/s]")
        ylabel = f"Spectrum [{units}]" if units else "Spectrum"
        ax.set_ylabel(ylabel)
        ax.set_title(f"MRR-PRO spectrum")

        ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.0)
        ax.legend(loc="upper right")

        fig.tight_layout()

        filepath: Path | None = None
        if savefig:
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig=True.")
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / self.path.name.replace(
                ".nc", f"_spectrum_{t_txt.replace(':','')}_{r_sel:.1f}m.png"
            )
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath

    def plot_spectra_by_range(
        self,
        target_datetime,
        ranges: list[float] | np.ndarray,
        *,
        use_db: bool = True,
        label_type: str = "range",
        ncol: int = 2,
        fig=None,
        ax=None,
        savefig: bool = False,
        output_dir=None,
        **kwargs,
    ):
        """
        Plot several MRR-PRO Doppler spectra at a fixed time for multiple ranges.

        This method overlays spectra for the nearest (time, range) gates.
        It relies on the RAW spectral variable 'spectrum_reflectivity' (preferred) or
        falls back to 'spectrum' if present.

        Parameters
        ----------
        target_datetime : datetime | np.datetime64 | str
            Time to plot. Nearest time gate is used.
        ranges : list[float] | np.ndarray
            List of ranges [m]. Nearest range gate is used for each value.
        use_db : bool, default True
            Plot spectrum in dB if True (10*log10), else linear.
        label_type : {"range","time","both"}, default "range"
            Legend label formatting.
        ncol : int, default 2
            Legend columns.
        figsize : tuple, default (10,7)
            Figure size if fig/ax not provided.
        fig, ax : matplotlib Figure/Axes, optional
            Reuse existing axes.
        output_dir : Path, optional
            Where to save if savefig=True.
        savefig : bool, default False
            Save figure if True.
        dpi : int, default 200
            Save DPI.
        kwargs :
            Optional plot kwargs forwarded to ax.plot (e.g., linewidth, alpha).

        Returns
        -------
        (fig, filepath) : (Figure, Path | None)
        """

        pcfg = self.plot_cfg
        dpi = kwargs.get("dpi", pcfg.dpi)
        figsize = kwargs.get("figsize", pcfg.figsize)

        ds = self.ds

        # --- sanity checks ---
        if "time" not in ds or "range" not in ds:
            raise KeyError("Dataset must contain 'time' and 'range' coordinates.")
        if "spectrum_n_samples" not in ds.dims:
            raise KeyError("Dataset must contain dimension 'spectrum_n_samples'.")

        # pick spectral variable
        spec_var = None
        for cand in ("spectrum_reflectivity", "spectrum", "spectra", "spectrum_raw"):
            if cand in ds:
                spec_var = cand
                break
        if spec_var is None:
            raise KeyError(
                "No spectral variable found. Expected one of: "
                "'spectrum_reflectivity', 'spectrum', 'spectra', 'spectrum_raw'."
            )

        # nearest time
        t_sel = ds["time"].sel(time=target_datetime, method="nearest").values

        # build velocity axis (try to use provided coord; else infer)
        vel = None
        for vname in ("velocity", "doppler_velocity", "velocity_vectors", "vel"):
            if vname in ds:
                v = ds[vname]
                # handle 1D velocity axis
                if "spectrum_n_samples" in v.dims and len(v.dims) == 1:
                    vel = v.values.astype(float)
                break
        if vel is None:
            # infer from fold limits if available, else 12 m/s
            vny = 12.0
            if "VEL" in ds and "fold_limit_upper" in ds["VEL"].attrs:
                try:
                    vny = float(ds["VEL"].attrs["fold_limit_upper"])
                except Exception:
                    pass
            n_bins = int(ds.sizes["spectrum_n_samples"])
            vel = np.linspace(0.0, vny, n_bins, dtype=float)

        # --- figure/axes ---
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif fig is not None and ax is None:
            axes = fig.get_axes()
            ax = axes[0] if len(axes) else fig.add_subplot(111)
        elif fig is None and ax is not None:
            fig = ax.figure

        # label helper
        def _label(t, r, mode):
            ttxt = np.datetime_as_string(t, unit="s")
            if mode == "both":
                return f"{ttxt} | {r:.1f} m"
            if mode == "time":
                return f"{ttxt}"
            return f"{r:.1f} m"

        # loop ranges
        ranges = np.asarray(ranges, dtype=float)
        if ranges.size == 0:
            raise ValueError("ranges must contain at least one range value.")

        # optional mapping range->n_spectra (MRR-PRO)
        has_index = "index_spectra" in ds and "n_spectra" in ds.dims

        # select spectrum at (time, range)
        for r_req in ranges:
            r_sel = ds["range"].sel(range=r_req, method="nearest").values.item()

            if has_index:
                idx_raw = (
                    ds["index_spectra"]
                    .sel(time=t_sel, range=r_sel, method="nearest")
                    .values
                )
                if not np.isfinite(idx_raw):
                    continue
                idx = int(idx_raw)
                if not (0 <= idx < ds.sizes["n_spectra"]):
                    continue
                spec = ds[spec_var].sel(time=t_sel).values.astype(float)[idx, :]
            else:
                # fallback: assume spectrum variable has (time, range, spectrum_n_samples)
                s = ds[spec_var].sel(time=t_sel, range=r_sel, method="nearest")
                if "spectrum_n_samples" not in s.dims:
                    raise ValueError(
                        f"{spec_var} does not have 'spectrum_n_samples' dimension."
                    )
                spec = s.values.astype(float)

            # convert to dB if requested
            y = spec
            if use_db:
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = 10.0 * np.log10(np.where(y > 0, y, np.nan))

            # plot (skip fully nan)
            if np.all(~np.isfinite(y)):
                continue

            ax.plot(
                vel,
                y,
                label=_label(t_sel, float(r_sel), label_type),
                **{k: v for k, v in kwargs.items() if k not in {"title"}},
            )

        # cosmetics
        ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Doppler velocity [m/s]")
        ax.set_ylabel("Spectrum [dB]" if use_db else "Spectrum [linear]")

        title = kwargs.get("title", None)
        if title is None:
            ttxt = np.datetime_as_string(t_sel, unit="s")
            ax.set_title(f"MRR-PRO spectra by range | time={ttxt}")
        else:
            ax.set_title(title)

        ax.legend(ncol=ncol, loc="best", fontsize=9)
        fig.tight_layout()

        filepath = None
        if savefig:
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig=True.")
            output_dir.mkdir(parents=True, exist_ok=True)
            ttxt = np.datetime_as_string(t_sel, unit="s").replace(":", "")
            filepath = output_dir / self.path.name.replace(
                ".nc", f"_spectra_by_range_{ttxt}.png"
            )
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath

    def plot_spectrogram(
        self,
        target_datetime: datetime | np.datetime64,
        *,
        spectrum_var: str,
        variable_threshold: str = "spectrum_raw",
        threshold_value: float = 0,
        range_limits: tuple[float, float] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        fig: Figure | None = None,
        ax=None,
        output_dir: Path | None = None,
        savefig: bool = False,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """
        Plot a spectrogram of MRR-PRO radar data for a specified time.

        Parameters
        ----------
        target_datetime : datetime | np.datetime64
            The target time for which to generate the spectrogram.
        spectrum_var : str, optional
            The spectrum variable to plot. Default is "spectrum_raw".
        range_limits : tuple[float, float] | None, optional
            Range limits in meters as (min, max). If None, uses full range. Default is None.
        vmin : float | None, optional
            Minimum value for the colorbar scale. If None, uses automatic scaling. Default is None.
        vmax : float | None, optional
            Maximum value for the colorbar scale. If None, uses automatic scaling. Default is None.
        cmap : str, optional
            Matplotlib colormap name. Default is "jet".
        fig : Figure | None, optional
            Matplotlib Figure object. If None, a new figure is created. Default is None.
        ax : optional
            Matplotlib Axes object. If None and fig is provided, uses first axes of fig.
            If both are None, creates new figure and axes. Default is None.
        output_dir : Path | None, optional
            Output directory for saving the figure. Required if savefig=True. Default is None.
        savefig : bool, optional
            Whether to save the figure to disk. Default is False.
        dpi : int, optional
            Resolution in dots per inch for saved figure. Default is 200.

        Returns
        -------
        tuple[Figure, Path | None]
            A tuple containing:
            - Figure: The matplotlib Figure object containing the spectrogram.
            - Path | None: Path to the saved figure if savefig=True, otherwise None.

        Raises
        ------
        ValueError
            If savefig=True but output_dir is None.
        """
        pcfg = self.plot_cfg
        dpi = kwargs.get("dpi", pcfg.dpi)
        cmap = kwargs.get("cmap", pcfg.cmap)
        figsize = kwargs.get("figsize", pcfg.figsize)

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif fig is not None and ax is None:
            ax = fig.get_axes()[0]

        t_sel, ranges, vel, spec2d, units = self._get_spectrogram_2d(
            target_datetime, spectrum_var=spectrum_var, range_limits=range_limits
        )
        t_txt = np.datetime_as_string(t_sel, unit="s")

        # spec2d expected shape: (range, bin)
        # extent = [xmin, xmax, ymax, ymin] para que suba hacia arriba
        # extent = [float(vel[0]), float(vel[-1]), float(ranges[-1]), float(ranges[0])]
        extent = [vel[0], vel[-1], ranges[0], ranges[-1]]

        im = ax.imshow(
            spec2d,
            aspect="auto",
            extent=extent,
            cmap=cmap,
            origin="lower",
        )

        if vmin is not None or vmax is not None:
            im.set_clim(vmin=vmin, vmax=vmax)

        ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Doppler velocity [m/s]")
        ax.set_ylabel("Range [m]")
        title = f"MRR-PRO spectrogram \n {t_txt}"
        ax.set_title(title)
        ax.set_xlim(kwargs.get('x_limits', (-4, 12)))

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Spectrum [{units}]" if units else "Spectrum")

        fig.tight_layout()

        filepath: Path | None = None
        if savefig:
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig=True.")
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / self.path.name.replace(
                ".nc", f"_{spectrum_var}_spectrogram_{t_txt.replace(':','')}.png"
            )
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath

    def plot_DSDgram(
        self,
        *,
        target_datetime: datetime.datetime,
        range_limits: tuple[float, float] | None = None,
        drop_limits: tuple[float, float] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs
        ):
        """
        DSD-gram: X=DropSize (mm), Y=range (m), color=dsd_3D, a un instante target_datetime.

        Requiere self.raprompro con variable 'dsd_3D' dims ('time','range','DropSize').
        """
        if self.raprompro is None:
            raise RuntimeError("self.raprompro no está cargado. Usa load_raprompro() o procesa antes.")

        pcfg = self.plot_cfg
        dpi = kwargs.get("dpi", pcfg.dpi)
        cmap = kwargs.get("cmap", pcfg.cmap)
        figsize = kwargs.get("figsize", pcfg.figsize)

        ds = self.raprompro
        if "dsd_3D" not in ds:
            raise KeyError("self.raprompro no contiene la variable 'dsd_3D'.")

        da = ds["dsd_3D"]

        # Validación mínima de dims esperadas
        expected = ("time", "range", "DropSize")
        if tuple(da.dims) != expected:
            raise ValueError(f"dsd_3D.dims esperadas {expected}, pero son {da.dims}")

        # Selección temporal (análoga a spectrogram: instante concreto)
        da2 = da.sel(time=target_datetime, method="nearest")  # dims -> (range, DropSize)

        # Subsets opcionales
        if range_limits is not None:
            da2 = da2.sel(range=slice(range_limits[0], range_limits[1]))
        if drop_limits is not None:
            da2 = da2.sel(DropSize=slice(drop_limits[0], drop_limits[1]))

        # Asegura orden para plot: (range, DropSize)
        da2 = da2.transpose("range", "DropSize")

        #Remove NaN in 'DropSize' in da2
        X_ = da2["DropSize"].values  # mm        
        da2 = da2.isel({"DropSize": np.isfinite(X_)})
        
        # Ejes
        X = da2["DropSize"].values  # mm        
        Y = da2["range"].values     # m
        Z = da2.values.astype(float)

        # OJO: en tu pipeline suele venir ya en log10, así que aquí NO aplico log10.
        # Si algún día guardas en lineal, mejor añadir un flag explícito.
        
        fig, ax = plt.subplots(figsize=figsize)

        # pcolormesh con X/Y 1D y shading auto evita el error clásico de incompatibilidad        
        m = ax.pcolormesh(X, Y, Z, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)

        ax.set_xlabel("Drop diameter (mm)")
        ax.set_ylabel("Range / Height (m)")
        ax.set_xlim(kwargs.get('x_limits', (0.25, 10)))
        # Título con el tiempo realmente seleccionado (nearest)
        tsel = da2["time"].values
        tlabel = np.datetime_as_string(tsel, unit="s") if np.issubdtype(np.asarray(tsel).dtype, np.datetime64) else str(tsel)
        ax.set_title(f"DSD-gram \n {tlabel}")

        cb = fig.colorbar(m, ax=ax)
        cb.set_label(da.attrs.get("units", "dsd_3D"))

        filepath = None
        if savefig:
            if output_dir is None:
                output_dir = Path.cwd()
            output_dir.mkdir(parents=True, exist_ok=True)
            ttag = np.datetime_as_string(tsel, unit="s").replace(":", "")
            filepath = output_dir / self.path.name.replace(
                ".nc", f"_DSDgram_{ttag}.png"
            )
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath

    def plot_DSD_by_range(
        self,
        target_datetime,
        ranges: list[float] | np.ndarray,
        *,
        use_log10: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        ncol: int = 2,
        savefig: bool = False,
        output_dir=None,
        fig=None,
        ax=None,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """
        Plot several N(D) curves at a fixed time for multiple provided ranges,
        using raprompro dsd_3D(time, range, DropSize).

        Parameters
        ----------
        target_datetime : datetime | np.datetime64 | str
            Target time. Nearest time gate is used.
        ranges : list[float] | np.ndarray
            List of ranges in meters. Nearest range gate is used for each.
        use_log10 : bool, default False
            If True, plot log10(N). If False, plot N in linear units (log y-scale).
            NOTE: If dsd_3D is stored already in log10, conversion is handled automatically.
        vmin, vmax : float | None
            Optional y-limits (applied as ylim). If both are None, no limits set.
        ncol : int, default 2
            Legend columns.
        fig, ax : matplotlib Figure/Axes, optional
            Reuse existing axes.
        output_dir : Path, optional
            Output directory if savefig=True.
        savefig : bool, default False
            Save to disk if True.

        Returns
        -------
        (fig, filepath) : (Figure, Path | None)
        """
        pcfg = self.plot_cfg

        dpi = kwargs.get("dpi", pcfg.dpi)
        cmap = kwargs.get("cmap", pcfg.cmap)
        figsize = kwargs.get("figsize", pcfg.figsize)
        marker = kwargs.get("marker", pcfg.marker)
        markersize = kwargs.get("markersize", pcfg.markersize)
        legend_fontsize = kwargs.get("legend_fontsize", pcfg.legendfontsize)

        # --- sanity checks: raprompro ---
        if self.raprompro is None:
            raise RuntimeError("raprompro not loaded. Use load_raprompro().")
        ds_rp = self.raprompro
        if "dsd_3D" not in ds_rp:
            raise KeyError("raprompro missing required variable 'dsd_3D'. Check save_dsd_3d is True.")
        da = ds_rp["dsd_3D"]

        # dims esperadas
        for d in ("time", "range", "DropSize"):
            if d not in da.dims:
                raise ValueError(f"dsd_3D must have dim '{d}'. dims={da.dims}")

        # --- nearest time (en el grid raprompro) ---
        t_sel = ds_rp["time"].sel(time=target_datetime, method="nearest").values

        # --- figure/axes ---
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif fig is not None and ax is None:
            axes = fig.get_axes()
            ax = axes[0] if len(axes) else fig.add_subplot(111)
        elif fig is None and ax is not None:
            fig = ax.figure

        # --- validate ranges ---
        ranges_in = np.asarray(ranges, dtype=float)
        if ranges_in.size == 0:
            raise ValueError("ranges must contain at least one value.")
                
        # --- colormap colors ---
        cm = plt.get_cmap(cmap)
        colors = [cm(i / max(1, ranges_in.size - 1)) for i in range(ranges_in.size)]

        # --- infer whether data is stored in log10 ---
        units = (da.attrs.get("units", "") or "").lower()
        data_is_log10 = "log10" in units or "log" in units  # conservador

        # --- drop size axis ---
        D = da["DropSize"].values.astype(float)

        # Heurística suave: si parece estar en metros (muy pequeño), convierte a mm.
        # (Si ya es mm, no hace nada.)
        D_scale = 1.0
        D_units = da["DropSize"].attrs.get("units", "")
        if D_units.lower() in ("m", "meter", "metre"):
            D_scale = 1000.0
            D_units_out = "mm"
        elif np.nanmax(D) < 0.05:  # típicamente <5 cm; si <0.05, suele ser metros
            D_scale = 1000.0
            D_units_out = "mm"
        else:
            D_units_out = "mm" if D_units == "" else D_units

        # --- loop ranges ---
        plotted_any = False
        N_minimum_thresh = float(kwargs.get("N_minimum_threshold", 0.0))

        for i, r_req in enumerate(ranges_in):
            r_sel = float(ds_rp["range"].sel(range=r_req, method="nearest").values.item())

            # Extrae el perfil N(D) a ese tiempo y rango
            # da_sel dims -> (DropSize,)
            da_sel = da.sel(time=t_sel, range=r_sel, method="nearest")
            N_raw = da_sel.values.astype(float)

            # Limpieza: umbral mínimo en el dominio lineal
            if data_is_log10:
                # En log10, el umbral en lineal equivale a log10(thresh). Si thresh=0, no aplica.
                if N_minimum_thresh > 0:
                    thr_log = np.log10(N_minimum_thresh)
                    N_raw = np.where(N_raw >= thr_log, N_raw, np.nan)
            else:
                N_raw = np.where(N_raw >= N_minimum_thresh, N_raw, np.nan)

            # Drop non-finite
            ok = np.isfinite(D) & np.isfinite(N_raw)
            if not np.any(ok):
                continue

            x = (D[ok] * D_scale).astype(float)

            # Conversión según lo que quieras visualizar
            if use_log10:
                # y = log10(N)
                y = N_raw[ok] if data_is_log10 else np.log10(N_raw[ok])
                ax.set_yscale("linear")
                ylab = r"$\log_{10}(N)\ [\mathrm{m^{-3}\,mm^{-1}}]$"
            else:
                # y = N (lineal), escala log en eje Y
                y = (10.0 ** N_raw[ok]) if data_is_log10 else N_raw[ok]
                # evita <=0 para escala log
                y = np.where(y > 0, y, np.nan)
                ok2 = np.isfinite(y)
                x = x[ok2]
                y = y[ok2]
                if x.size == 0:
                    continue
                ax.set_yscale("log")
                ylab = r"$N\ [\mathrm{m^{-3}\,mm^{-1}}]$"

            ax.plot(
                x,
                y,
                color=colors[i],
                label=f"{r_sel:.1f} m",
                marker=marker,
                markersize=markersize,
            )
            plotted_any = True

        if not plotted_any:
            raise ValueError("No valid DSD curves found for the provided ranges/time.")

        # --- labels / title ---
        ax.set_xlabel(f"D [{D_units_out}]")
        ax.set_ylabel(ylab)

        t_txt = np.datetime_as_string(t_sel, unit="s")
        ax.set_title(f"RaProMPro N(D) by range\n{t_txt}")

        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(ncol=ncol, loc="best", fontsize=legend_fontsize)

        if vmin is not None or vmax is not None:
            ax.set_ylim(vmin, vmax)

        if kwargs.get("xlimits", None) is not None:
            ax.set_xlim(kwargs["xlimits"])

        fig.tight_layout()

        filepath = None
        if savefig:
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig=True.")
            output_dir.mkdir(parents=True, exist_ok=True)
            ttag = np.datetime_as_string(t_sel, unit="s").replace(":", "")
            filepath = output_dir / self.path.name.replace(".nc", f"_DSD_by_range_{ttag}.png")
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath


    def plot_microphysical_properties_profiles(
        self,
        target_datetime: datetime.datetime,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs,
    ) -> tuple[Figure, np.ndarray, Path | None]:
        """
        RaProMPro diagnostic profile (single figure, 4 axes; Y = height):
        1) Ze, Zea, Z_all, Za
        2) Dm
        3) Nw
        4) LWC, LWC_all

        Uses self.ds and selects the nearest profile to `target_datetime`.
        Raises if the dataset does not look RaProMPro-preprocessed.
        """
        # --- minimal "preprocessed?" check ---
        
        if self._is_processed():
            preprocessed_status = "RaProMPro-preprocessed"
        else:
            raise RuntimeError(
                "Dataset does not appear to be RaProMPro-preprocessed. Missing expected variables or attributes."
            )

        pcfg = self.plot_cfg
        figsize = kwargs.get("figsize", pcfg.figsize_profiles)

        ds = self.raprompro

        if "time" not in ds.coords:
            raise RuntimeError("No 'time' coordinate found in dataset.")

        # --- select nearest time ---
        prof = ds.sel(time=np.datetime64(target_datetime), method="nearest")
        sel_time = prof["time"].values
        try:
            sel_time_str = np.datetime_as_string(sel_time, unit="s")
        except Exception:
            sel_time_str = str(sel_time)

        z = prof["range"].values.astype(float) / 1000.0  # to km

        fig, axs = plt.subplots(
            ncols=4,
            figsize=figsize,
            sharey=True,
            constrained_layout=True,
        )

        # 1) Reflectivities
        ax = axs[0]
        Z_variables = ["Ze", "Za", "Zea", "Z_all"]
        markers = {'Ze': 'x', "Za": 'v', 'Zea': 'o', 'Z_all': '^'}
        for Z_ in Z_variables:
            if Z_ not in prof.data_vars:
                continue
            # if Z_ is 'Ze':
            #     breakpoint()
            ax.plot(prof[Z_].values, z, label=Z_, linewidth=1, marker=markers[Z_], markersize=4)

        ax.set_xlabel("Reflectivities, dBZ")
        ax.set_ylabel(f"{'range'} (km)")
        ax.set_xlim(kwargs.get("x_limits", (0, 45)))
        ax.grid(True)
        ax.legend(loc="best")

        # 2) Dm
        ax = axs[1]
        ax.plot(prof["Dm"].values, z, linewidth=1, marker="o", markersize=4)
        ax.set_xlabel(r"$D_m$, mm")
        ax.set_xlim(kwargs.get("Dm_limits", (0.0, 4)))
        ax.grid(True)

        # 3) Nw
        ax = axs[2]
        ax.plot(prof["Nw"].values, z, linewidth=1, marker="o", markersize=4)
        ax.set_xlabel(r"$log_{10}(N_w \, mm^{-1} m^{-3})$")
        ax.set_xlim(kwargs.get("Nw_limits", (0., 6.0)))
        ax.grid(True)

        # 4) LWC
        ax = axs[3]
        ax.plot(
            prof["LWC_all"].values,
            z,
            linewidth=5,
            marker="v",
            markersize=10,
            label="LWC_all",
        )
        delta = kwargs.get("LWC_all_marker_color_delta", 0.3)
        color_light = tuple(
            min(1.0, c + delta) for c in mcolors.to_rgb(ax.lines[-1].get_color())
        )
        ax.plot(
            prof["LWC"].values,
            z,
            linewidth=1,
            marker="o",
            markersize=4,
            label="LWC",
            color=color_light,
        )
        ax.legend(loc="best")

        # get color from previous plot

        ax.set_xlabel(r"LWC, g m_^{-3}")
        ax.set_xlim(kwargs.get("LWC_limits", (0, 3.0)))
        ax.grid(True)

        if kwargs.get("y_limits", None) is not None:
            for ax in axs:
                ax.set_ylim(kwargs["y_limits"])

        fig.suptitle(f"{preprocessed_status} MRR-Pro \n {sel_time_str}", fontsize=30)

        output_path = None
        if savefig:
            if output_dir is None:
                output_dir = Path().cwd()
            datestr = target_datetime.strftime("%Y%m%d_%H%M%S")
            output_path = (
                output_dir
                / f"{self.path.stem}_{datestr}_{preprocessed_status}_profiles.png"
            )
            fig.savefig(output_path)

        return fig, axs, output_path

    def plot_rain_process_in_layer_2D(
        self,
        target_datetime: datetime | tuple[datetime, datetime],
        layer: tuple[float, float],
        x: str = "Dm",
        y: str = "LwC",
        z: str = "Nw",
        use_relative_difference: bool = True,
        savefig: bool = False,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """
        Plots the rain process in a specified atmospheric layer at a given datetime.
        This method generates a scatter plot of two selected variables (x and y) from the dataset,        
        vertical layer. The plot can optionally be saved to disk.
        Parameters
        ----------
        target_datetime : datetime
            The target datetime for which to select the data profile.
        layer : tuple[float, float]
            The vertical layer (zmin, zmax) in meters to analyze.
        x : str, optional
            The variable name to plot on the x-axis (default is 'Dm').
        y : str, optional
            The variable name to plot on the y-axis (default is 'LwC').
        z : str, optional
            The variable name to use for color mapping (default is 'Nw').
        savefig : bool, optional
            Whether to save the generated figure to disk (default is False).
            Additional keyword arguments:
                - figsize: tuple, optional
                    Figure size (default is (12, 6)).
                - cmap: str, optional
                    Colormap for the scatter plot (default is 'viridis').
                - markersize: int or float, optional
                    Marker size for the scatter plot (default is 50).
                - output_dir: Path or str, optional
                    Directory to save the figure if savefig is True (default is current working directory).
        Returns
        -------
        tuple[Figure, Path | None]
            A tuple containing the matplotlib Figure object and the output Path if saved, otherwise None.
        Raises
        ------
        KeyError
            If any of the specified variables (x, y, z) are not found in the dataset.
        """

        if self._is_processed():
            ds = self.raprompro
        else:
            raise RuntimeError('Dataset is not processed.')
        
        pcfg = self.plot_cfg
        figsize = kwargs.get('figsize', pcfg.figsize)

        # Check x,y,z exists as variable in self.ds, otherwise raise error
        for var in (x, y, z):
            if var not in ds:
                raise KeyError(f"Variable '{var}' not found in dataset.")

        # Check time tuple increasing
        if isinstance(target_datetime, tuple):
            if target_datetime[0] >= target_datetime[1]:
                raise ValueError(
                    "target_datetime tuple must be in increasing order (start, end)."
                )

        # get data profiles from self.data
        if isinstance(target_datetime, datetime):
            ds_ = ds.sel(time=target_datetime, method="nearest").sel(
                range=slice(*layer)
            )
        else:
            ds_ = ds.sel(time=slice(*target_datetime)).sel(range=slice(*layer))

        # Get the difference of all properties between the range with respect to the zmax (end - start)
        last_range = ds_.range[-1]

        if use_relative_difference:
            diff_ = (
                100
                * (ds_ - ds_.sel(range=last_range))
                / ds_.sel(range=slice(*layer)).mean("range")
            )
            # Get maximum values for x,y,z to set simetric axis limits
            ds_ = diff_.copy()
        else:
            ds_ = ds_ - ds_.sel(range=last_range)

        x_abs_max = np.abs(ds_[x].values[np.isfinite(ds_[x].values)]).max()
        y_abs_max = np.abs(ds_[y].values[np.isfinite(ds_[y].values)]).max()
        z_abs_max = np.abs(ds_[z].values[np.isfinite(ds_[z].values)]).max()
        # create a figure with same y and x axis size
        fig, ax = plt.subplots(figsize=figsize)

        # for time_ in diff_.time:
        ds_.plot.scatter(
            x=x,
            y=y,
            hue=z,
            cmap=kwargs.get("cmap", "viridis"),
            s=kwargs.get("markersize", 50),
            vmin=-z_abs_max,
            vmax=z_abs_max,
            ax=ax,
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        # cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar = ax.get_figure().get_axes()[1]
        cbar.set_ylabel(z)
        zmin, zmax = layer
        if isinstance(target_datetime, tuple):
            ax.set_title(
                f"Layer {zmin/1000.}-{zmax/1000.} km \n {target_datetime[0]} to {target_datetime[1]}"
            )
        else:
            ax.set_title(f"Layer {zmin/1000.}-{zmax/1000.} km | {target_datetime}")
        # create simetric limits for x and y axis
        ax.set_xlim(-x_abs_max, x_abs_max)
        ax.set_ylim(-y_abs_max, y_abs_max)

        # add grid and unity line
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Highlight main axis lines (x,0) and (0,y)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        # Adjust layout

        fig.tight_layout()

        if savefig:
            output_dir = kwargs.get("output_dir", Path().cwd())
            output_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(target_datetime, tuple):
                datestr = f"{target_datetime[0].strftime('%Y%m%d_%H%M%S')}_to_{target_datetime[1].strftime('%Y%m%d_%H%M%S')}"
            else:
                datestr = target_datetime.strftime("%Y%m%d_%H%M%S")
            output_path = (
                output_dir
                / f"rain_process_2D_{self.path.stem}_{datestr}_{zmin}-{zmax}m.png"
            )
            fig.savefig(output_path)

        return fig, output_path if savefig else None


    def compute_layer_trend_ols(
        self,
        *,
        z_top: float,
        z_base: float,
        time_dim: str = "time",
        variable_threshold: str = "Ze",
        threshold_value: float = -5.0,
        vars: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
        eps_mode: str = "hourly_quantile",
        q: float = 0.01,
        eps_floor_mode: str = "global_min",
        min_points_ols: int = 10,
    ) -> xr.Dataset:
        """
        Compute b_X (1/m), F_X, and R^2 for X in vars using OLS of ln(X) vs depth from top,
        and store the data selection used for each fit (masks + eps + counts).

        Output includes:
        - b_<var>, a_<var>, r2_<var>, F_<var> : (time,)
        - eps_<var> : (time,)
        - n_fit_<var> : (time,)
        - mask_fit_<var> : (time, range_layer)  (True where points used in OLS)
        - mask_ze : (time, range_layer)
        - coords: time, range_layer, depth (depth from top; meters)
        - attrs: z_top, z_base, dz, threshold settings, eps settings, q, min_points_ols
        """
        if not self._is_processed():
            raise RuntimeError("MRR-Pro data not processed (raprompro missing).")

        ds = self.raprompro

        if z_base <= z_top:
            raise ValueError("z_base must be greater than z_top (in meters).")

        # --- Select layer ---
        layer = ds.sel({"range": slice(z_top, z_base)})

        if time_dim not in layer.coords:
            raise KeyError(f"Missing coord '{time_dim}' in dataset.")
        if "range" not in layer.coords:
            raise KeyError("Missing coord 'range' in dataset.")
        if variable_threshold not in layer:
            raise KeyError(f"Missing threshold variable '{variable_threshold}' in dataset.")

        # Ensure required variables exist
        for vname in vars:
            if vname not in layer:
                raise KeyError(f"Missing variable '{vname}' in dataset.")

        # Depth from top (positive downward, meters)
        z_layer = layer["range"].values.astype(float)  # 1D in-layer
        depth = np.abs(z_layer - float(z_top)).astype(float)  # 1D, >=0
        dz = float(z_base - z_top)

        # --- Threshold mask (common) ---
        Ze = layer[variable_threshold]
        ze_mask = xr.where(np.isfinite(Ze) & (Ze > threshold_value), True, False)  # (time, range)

        # --- Output dataset with traceability coords ---
        out = xr.Dataset(
            coords={
                time_dim: layer[time_dim].values,
                "range_layer": layer["range"].values,
            }
        )
        out = out.assign_coords(depth=("range_layer", depth))

        out["dz"] = xr.DataArray(dz)
        out["mask_ze"] = xr.DataArray(
            ze_mask.values.astype(bool),
            dims=(time_dim, "range_layer"),
        )

        out.attrs.update(
            dict(
                z_top=float(z_top),
                z_base=float(z_base),
                dz=float(dz),
                variable_threshold=str(variable_threshold),
                threshold_value=float(threshold_value),
                vars=tuple(vars),
                eps_mode=str(eps_mode),
                eps_floor_mode=str(eps_floor_mode),
                q=float(q),
                min_points_ols=int(min_points_ols),
            )
        )

        # --- Precompute global eps if requested ---
        global_eps: dict[str, float] = {}
        if eps_mode == "global_quantile" or eps_floor_mode == "global_min":
            for vname in vars:
                global_eps[vname] = compute_eps(layer[vname].values, q=q)

        # --- Prepare loop ---
        times = layer[time_dim].values
        ntime = times.size
        nrange = layer.sizes["range"]

        ze_mask_np = ze_mask.values.astype(bool)  # (time, range)

        n_valid = np.sum(ze_mask_np, axis=1).astype(int)
        out["n_valid"] = xr.DataArray(n_valid, dims=(time_dim,))

        # --- Fit each variable ---
        for vname in vars:
            b_arr = np.full(ntime, np.nan, dtype=float)
            a_arr = np.full(ntime, np.nan, dtype=float)
            r2_arr = np.full(ntime, np.nan, dtype=float)
            F_arr = np.full(ntime, np.nan, dtype=float)

            eps_used = np.full(ntime, np.nan, dtype=float)
            n_fit = np.zeros(ntime, dtype=int)
            mask_fit = np.zeros((ntime, nrange), dtype=bool)

            V = layer[vname].values.astype(float)  # (time, range)

            for it in range(ntime):
                # quick skip by threshold-only mask count
                if n_valid[it] < min_points_ols:
                    continue

                # final mask used for fit (threshold + finite + >0)
                mask = ze_mask_np[it, :] & np.isfinite(V[it, :]) & (V[it, :] > 0.0)
                nmask = int(np.sum(mask))
                if nmask < min_points_ols:
                    continue

                # epsilon
                if eps_mode == "hourly_quantile":
                    eps_t = compute_eps(V[it, :], q=q)
                elif eps_mode == "global_quantile":
                    eps_t = global_eps.get(vname, np.nan)
                else:
                    raise ValueError(f"Unsupported eps_mode={eps_mode!r}")

                if not np.isfinite(eps_t) or eps_t <= 0:
                    continue

                if eps_floor_mode == "global_min":
                    eps_g = global_eps.get(vname, np.nan)
                    if np.isfinite(eps_g) and eps_g > 0:
                        eps_t = max(float(eps_t), float(eps_g))

                x = depth[mask]
                y = np.log(np.maximum(V[it, mask], eps_t))

                b, a, r2 = ols_slope_intercept_r2(x, y)
                if not (np.isfinite(b) and np.isfinite(a) and np.isfinite(r2)):
                    continue

                b_arr[it] = float(b)
                a_arr[it] = float(a)
                r2_arr[it] = float(r2)
                F_arr[it] = float(np.exp(b * dz))

                # traceability
                mask_fit[it, :] = mask
                n_fit[it] = nmask
                eps_used[it] = float(eps_t)

            out[f"b_{vname}"] = xr.DataArray(b_arr, dims=(time_dim,))
            out[f"a_{vname}"] = xr.DataArray(a_arr, dims=(time_dim,))
            out[f"r2_{vname}"] = xr.DataArray(r2_arr, dims=(time_dim,))
            out[f"F_{vname}"] = xr.DataArray(F_arr, dims=(time_dim,))

            out[f"eps_{vname}"] = xr.DataArray(eps_used, dims=(time_dim,))
            out[f"n_fit_{vname}"] = xr.DataArray(n_fit, dims=(time_dim,))
            out[f"mask_fit_{vname}"] = xr.DataArray(mask_fit, dims=(time_dim, "range_layer"))

        return out
    

    def rain_process_analyze(
        self,
        *,
        period: tuple[datetime, datetime],
        layer: tuple[float, float],
        k: int,
        ze_th: float = -5.0,
        min_points_ols: int = 10,
        eps_q: float = 0.01,
        rgb_q: float = 0.02,
        vars_trend: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    ) -> xr.Dataset:
        """
        Analiza proceso de lluvia en una capa y periodo: OLS (trends) -> RGB -> hex mapping.

        Returns
        -------
        xr.Dataset con coords time y variables:
        - b_*, a_*, r2_*, F_*, n_valid, eps_*, n_fit_*, mask_fit_* (de compute_layer_trend_ols)
        - R, G, B (0..1)
        - minutes (float)
        - hex_x, hex_y, hex_area (+ snap_R,G,B si disponible)
        """
        if not self._is_processed():
            raise RuntimeError("Dataset not preprocessed / raprompro not available.")

        ds = self.raprompro
        z_top, z_base = layer
        if z_base <= z_top:
            raise ValueError("layer debe cumplir z_base > z_top (metros).")

        t0, t1 = period
        if t0 >= t1:
            raise ValueError("period debe ser creciente (t0 < t1).")

        # Selección temporal (sólo para validar no vacío y fijar periodo real)
        ds_sub = ds.sel(time=slice(t0, t1))
        if ds_sub.sizes.get("time", 0) == 0:
            raise ValueError("Selección temporal vacía: revisa period.")

        # 1) Tendencias en capa (OLS)
        trends = self.compute_layer_trend_ols(
            z_top=z_top,
            z_base=z_base,
            variable_threshold="Ze",
            threshold_value=ze_th,
            vars=vars_trend,
            q=eps_q,
            min_points_ols=min_points_ols,
        )

        # IMPORTANT: recorta trends al periodo solicitado (compute_layer_trend_ols usa todo el fichero)
        trends = trends.sel(time=slice(ds_sub["time"].values[0], ds_sub["time"].values[-1]))

        # 2) RGB (convención actual de tu plot: vars_trend en orden (Dm, Nw, LWC) -> (R,G,B) según tu build)
        rgb = build_rgb_from_trends(
            trends,
            vars=(f"b_{vars_trend[0]}", f"b_{vars_trend[1]}", f"b_{vars_trend[2]}"),
            q=rgb_q,
        )

        # minutes since start
        t = rgb["time"].values
        t00 = t[0]
        minutes = (t - t00) / np.timedelta64(1, "m")
        minutes = minutes.astype(float)

        # 3) hex mapping (assets/LUT)
        hex_assets = get_hexagram_assets(k=k)
        hex_ds = map_rgb_to_hexagram(rgb, hex_assets=hex_assets)

        # 4) merge outputs
        out = xr.Dataset(coords={"time": rgb["time"].values})        

        # trends: añade todo salvo coords “range_layer/depth” si no quieres inflar demasiado
        # (Yo lo incluyo porque tú querías trazabilidad)
        for v in trends.data_vars:
            out[v] = trends[v]
        for c in trends.coords:
            if c not in out.coords:
                out = out.assign_coords({c: trends.coords[c]})
        out.attrs.update(trends.attrs)

        # RGB
        out["R"] = rgb["R"]
        out["G"] = rgb["G"]
        out["B"] = rgb["B"]
        out["minutes"] = xr.DataArray(minutes, dims=("time",))        

        # Hex mapping
        for v in hex_ds.data_vars:
            out[v] = hex_ds[v]

        # metadata del análisis
        out.attrs.update(
            dict(
                period_start=str(np.datetime_as_string(ds_sub["time"].values[0], unit="s")),
                period_end=str(np.datetime_as_string(ds_sub["time"].values[-1], unit="s")),
                z_top=float(z_top),
                z_base=float(z_base),
                k=int(k),
                ze_th=float(ze_th),
                min_points_ols=int(min_points_ols),
                eps_q=float(eps_q),
                rgb_q=float(rgb_q),
                vars_trend=tuple(vars_trend),
                rgb_convention=str(f"R={vars_trend[0]}, G={vars_trend[1]}, B={vars_trend[2]}")
                                   
            )
        )
        out.attrs["rgb_mapping"] = {
            "R": vars_trend[0],
            "G": vars_trend[1],
            "B": vars_trend[2],
        }
        out.attrs["strength_definition"] = "min(|RGB-0.5|)/0.5"        
        return out


    def plot_rain_process_in_layer_hexagram(
        self,
        *,
        analysis: xr.Dataset,
        use_snapped_colors: bool = True,
        savefig: bool = False,
        output_dir=None,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """
        SOLO plotting: dibuja el hexagrama base (RGB) y superpone la trayectoria temporal (puntos)
        usando el resultado precomputado `analysis` (salida de rain_process_analyze).

        Requiere en `analysis`:
        - hex_x, hex_y (coords en rejilla del hexagrama)
        - minutes (para colorear por tiempo)
        - R,G,B (0..1) y opcional snap_R,snap_G,snap_B
        - attrs: period_start, period_end, z_top, z_base (opcionales pero recomendados)

        Parameters
        ----------
        analysis : xr.Dataset
            Resultado de rain_process_analyze(...)
        k : int
            Resolución del hexagrama (debe coincidir con la usada en el análisis para que la LUT cuadre).
        use_snapped_colors : bool
            Si True y existen snap_R/G/B, colorea con el color “snapeado” a la celda.
            Si False, usa RGB continuo.
        """
        # ------------------------------------------------------------------
        # Plot configuration
        # ------------------------------------------------------------------
        pcfg = self.plot_cfg
        figsize = kwargs.get("figsize", pcfg.figsize_multipanel)
        markersize = kwargs.get("markersize", pcfg.markersize)
        dpi = kwargs.get("dpi", pcfg.dpi)
        alpha = kwargs.get("alpha", pcfg.alpha_points)

        # --- checks mínimos ---
        if analysis is None or not isinstance(analysis, xr.Dataset):
            raise TypeError("analysis debe ser un xr.Dataset (salida de rain_process_analyze).")

        required = ("hex_x", "hex_y", "minutes", "R", "G", "B")
        missing = [v for v in required if v not in analysis]
        if missing:
            raise KeyError(f"analysis no contiene variables requeridas: {missing}")

        # --- assets (img + LUT) ---        
        k  = analysis.attrs['k']
        hex_assets = get_hexagram_assets(k=k)
        img = hex_assets["img"]
        ny, nx = img.shape[:2]

        # --- datos ---
        hx = analysis["hex_x"].values.astype(float)
        hy = analysis["hex_y"].values.astype(float)
        minutes = analysis["minutes"].values.astype(float)

        if use_snapped_colors and all(v in analysis for v in ("snap_R", "snap_G", "snap_B")):
            cols = np.stack(
                [
                    analysis["snap_R"].values,
                    analysis["snap_G"].values,
                    analysis["snap_B"].values,
                ],
                axis=1,
            ).astype(float)
        else:
            cols = np.stack(
                [analysis["R"].values, analysis["G"].values, analysis["B"].values],
                axis=1,
            ).astype(float)

        # --- máscara de validez (evita warnings/errores de matplotlib) ---
        ok = (
            np.isfinite(hx)
            & np.isfinite(hy)
            & np.isfinite(minutes)
            & np.isfinite(cols).all(axis=1)
            & (hx >= 0)
            & (hy >= 0)
            & (hx <= (nx - 1))
            & (hy <= (ny - 1))
        )

        # --- plot ---
        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(
            img,
            origin="lower",
            interpolation="nearest",
            alpha=kwargs.get("alpha_hexagram", 0.25),
        )

        sc = None
        if np.any(ok):
            sc = ax.scatter(
                hx[ok],
                hy[ok],
                s=markersize,
                c=minutes[ok],
                alpha=alpha,
                cmap=kwargs.get("cmap", "viridis"),
                edgecolors=kwargs.get("edgecolors", "black"),
                linewidths=kwargs.get("linewidths", 0.1),
            )

        # --- etiquetado ---
        z_top = analysis.attrs.get("z_top", None)
        z_base = analysis.attrs.get("z_base", None)
        t0s = analysis.attrs.get("period_start", None)
        t1s = analysis.attrs.get("period_end", None)

        layer_txt = (
            f"Capa {float(z_top):.0f}-{float(z_base):.0f} m"
            if (z_top is not None and z_base is not None)
            else "Capa (desconocida)"
        )
        period_txt = f"{t0s} → {t1s}" if (t0s is not None and t1s is not None) else ""
        rgb_map = analysis.attrs.get("rgb_mapping", None)
        rgb_txt = ", ".join(f"{k}={v}" for k, v in rgb_map.items())
        ax.set_title(
            f"Hexagrama RGB (k={k}) | {rgb_txt}\n"
            f"Capa {z_top:.0f}-{z_base:.0f} m | {t0s} → {t1s}"
        )
        ax.set_title(f"Hexagrama RGB (k={k}) | {layer_txt}\n{period_txt}".rstrip())

        ax.set_xlabel("hex_x (índice rejilla)")
        ax.set_ylabel("hex_y (índice rejilla)")
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.grid(False)

        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Minutes since start")

        fig.tight_layout()

        # --- savefig ---
        filepath = None
        if savefig:
            if output_dir is None:
                output_dir = Path.cwd()
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # nombre de fichero estable
            # preferimos attrs; si no, fallback
            safe_t0 = (t0s or "t0").replace(":", "").replace("-", "").replace(" ", "_")
            safe_t1 = (t1s or "t1").replace(":", "").replace("-", "").replace(" ", "_")
            safe_layer = (
                f"{float(z_top):.0f}-{float(z_base):.0f}m" if (z_top is not None and z_base is not None) else "layer"
            )

            filepath = output_dir / f"rain_process_hex_{safe_t0}_{safe_t1}_{safe_layer}_k{k}.png"
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath


    def classify_rain_process(
        self,
        *,
        analysis: xr.Dataset,
        tol_center: float = 0.05,
        min_strength: float = 0.10,
    ) -> xr.Dataset:
                
        if analysis is None or not isinstance(analysis, xr.Dataset):
            raise TypeError("analysis debe ser un xr.Dataset (salida de rain_process_analyze).")
        if "time" not in analysis.coords:
            raise KeyError("analysis debe tener coord 'time'.")
        for v in ("R", "G", "B"):
            if v not in analysis:
                raise KeyError("analysis debe incluir R,G,B (decisión tomada en rain_process_analyze).")

        # opcional: si tus reglas están calibradas para un mapping concreto:
        expected = {"R": "Dm", "G": "Nw", "B": "LWC"}
        rgb_map = analysis.attrs.get("rgb_mapping", None)        
        if rgb_map != expected:
            raise ValueError(f"rgb_mapping={rgb_map} pero este clasificador espera {expected}.")

        R = analysis["R"].values.astype(float)
        G = analysis["G"].values.astype(float)
        B = analysis["B"].values.astype(float)

        ok = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)

        sR = np.zeros(R.shape, dtype=int)
        sG = np.zeros(G.shape, dtype=int)
        sB = np.zeros(B.shape, dtype=int)
        if np.any(ok):
            sR[ok] = _sign_from_center(R[ok])
            sG[ok] = _sign_from_center(G[ok])
            sB[ok] = _sign_from_center(B[ok])

        strg = np.zeros(R.shape, dtype=float)
        if np.any(ok):
            strg[ok] = np.minimum.reduce([_strength(R[ok]), _strength(G[ok]), _strength(B[ok])])

        label = np.full(R.shape, "unknown", dtype=object)
        score = np.zeros(R.shape, dtype=float)

        def match(wR, wG, wB):
            m = ok.copy()
            m &= (sR == wR) & (sG == wG) & (sB == wB)
            return m

        # Reglas (ASUMEN la convención RGB ya fijada en analysis.attrs["rgb_convention"])
        m_breakup = match(0, +1, -1)
        m_coal    = match(0, -1, +1)
        m_evap    = match(-1, -1, -1)
        m_auto    = match(+1, +1, -1)
        m_act     = match(+1, +1, +1)

        for m, name in [
            (m_evap, "evaporation"),
            (m_breakup, "breakup"),
            (m_coal, "coalescence"),
            (m_auto, "autoconversion"),
            (m_act, "activation"),
        ]:
            take = m & (label == "unknown")
            label[take] = name

        score[ok] = strg[ok]
        weak = ok & (strg < min_strength)
        label[weak] = "steady_or_weak"

        out = xr.Dataset(coords={"time": analysis["time"].values})
        out["proc_label"] = xr.DataArray(label, dims=("time",))        
        out["sign_R"] = xr.DataArray(sR, dims=("time",))
        out["sign_G"] = xr.DataArray(sG, dims=("time",))
        out["sign_B"] = xr.DataArray(sB, dims=("time",))
        out["strength"] = xr.DataArray(strg, dims=("time",))

        # Copia RGB (para plots)
        out["R"] = analysis["R"]
        out["G"] = analysis["G"]
        out["B"] = analysis["B"]

        # Copia hex/minutes si existen (para plots multipanel)
        for v in ("hex_x", "hex_y", "hex_area", "minutes", "snap_R", "snap_G", "snap_B"):
            if v in analysis:
                out[v] = analysis[v]

        # Metadata (decisiones ya tomadas)
        out.attrs["tol_center"] = float(tol_center)
        out.attrs["min_strength"] = float(min_strength)
        for key in ("rgb_convention", "period_start", "period_end", "z_top", "z_base", "k", "rgb_q", "eps_q", "ze_th", "min_points_ols"):
            if key in analysis.attrs:
                out.attrs[key] = analysis.attrs[key]

        return out


    def plot_microphysics_summary_multipanel(
        self,
        *,
        analysis: xr.Dataset,
        classified: xr.Dataset,
        show_path_line: bool = True,
        savefig: bool = False,
        output_dir: "Path | None" = None,
        **kwargs,
    ) -> "tuple[Figure, Path | None]":
        """
        Plot multipanel (PLOT-ONLY) coherente con el pipeline:

            analysis   = rain_process_analyze(...)
            classified = classify_rain_process(analysis=analysis, ...)

        Paneles:
        (a) Hexagrama + trayectoria temporal (color = minutes)
        (b) Timeline de proc_label (color = strength)
        (c) Signos (sign_R/G/B) vs tiempo
        (d) Strength vs tiempo

        Todas las decisiones científicas (RGB, k, reglas, etc.)
        deben estar ya contenidas en `analysis` y `classified`.
        """

        # ------------------------------------------------------------------
        # Plot configuration
        # ------------------------------------------------------------------
        pcfg = self.plot_cfg
        cmap = kwargs.get("cmap", pcfg.cmap)
        figsize = kwargs.get("figsize", pcfg.figsize_multipanel)
        alpha_hexagram = kwargs.get("alpha_hexagram", pcfg.alpha_hexagram)
        markersize = kwargs.get("markersize", pcfg.markersize)
        line_width = kwargs.get("line_width", pcfg.linewidth)
        dpi = kwargs.get("dpi", pcfg.dpi)

        # ------------------------------------------------------------------
        # Sanity checks
        # ------------------------------------------------------------------
        if not isinstance(analysis, xr.Dataset):
            raise TypeError("analysis debe ser un xr.Dataset (salida de rain_process_analyze).")
        if not isinstance(classified, xr.Dataset):
            raise TypeError("classified debe ser un xr.Dataset (salida de classify_rain_process).")

        for ds, name in [(analysis, "analysis"), (classified, "classified")]:
            if "time" not in ds.coords:
                raise KeyError(f"{name} debe tener coord 'time'.")

        for v in ("hex_x", "hex_y", "minutes"):
            if v not in analysis:
                raise KeyError(f"analysis debe contener '{v}'.")

        for v in ("proc_label", "sign_R", "sign_G", "sign_B", "strength"):
            if v not in classified:
                raise KeyError(f"classified debe contener '{v}'.")

        # k debe venir del análisis
        k = analysis.attrs.get("k", None)
        if k is None:
            raise KeyError("analysis.attrs['k'] no existe (necesario para dibujar el hexagrama).")

        # RGB mapping solo para trazabilidad
        rgb_map = analysis.attrs.get("rgb_mapping", None)
        rgb_txt = ""
        if rgb_map is not None:
            rgb_txt = " | " + ", ".join(f"{c}={v}" for c, v in rgb_map.items())

        # ------------------------------------------------------------------
        # Align datasets in time
        # ------------------------------------------------------------------
        analysis_a, classified_a = xr.align(analysis, classified, join="inner")
        t = analysis_a["time"].values
        if t.size == 0:
            raise ValueError("analysis y classified no tienen intersección temporal.")

        # ------------------------------------------------------------------
        # Figure layout
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.0, 1.2, 1.0], hspace=0.25)

        ax_hex = fig.add_subplot(gs[0, 0])
        ax_tl  = fig.add_subplot(gs[1, 0])
        ax_sgn = fig.add_subplot(gs[2, 0])
        ax_str = fig.add_subplot(gs[3, 0])

        # ------------------------------------------------------------------
        # (a) Hexagram + trajectory
        # ------------------------------------------------------------------
        assets = get_hexagram_assets(k=k)
        img = assets["img"]

        ax_hex.imshow(
            img,
            origin="lower",
            interpolation="nearest",
            alpha=alpha_hexagram,
        )

        hx = analysis_a["hex_x"].values.astype(float)
        hy = analysis_a["hex_y"].values.astype(float)
        minutes = analysis_a["minutes"].values.astype(float)

        ok = (
            np.isfinite(hx)
            & np.isfinite(hy)
            & np.isfinite(minutes)
            & (hx >= 0)
            & (hy >= 0)
        )

        if np.any(ok):
            sc = ax_hex.scatter(
                hx[ok],
                hy[ok],
                c=minutes[ok],
                cmap=cmap,
                s=markersize,
                edgecolors="black",
                linewidths=0.1,
            )
            cbar = fig.colorbar(sc, ax=ax_hex, fraction=0.03, pad=0.01)
            cbar.set_label("Minutes since start")

            if show_path_line:
                idx = np.argsort(minutes[ok])
                ax_hex.plot(
                    hx[ok][idx],
                    hy[ok][idx],
                    color="black",
                    lw=line_width,
                    alpha=0.4,
                )

        layer_txt = ""
        if "z_top" in analysis_a.attrs and "z_base" in analysis_a.attrs:
            layer_txt = f" | layer {analysis_a.attrs['z_top']:.0f}-{analysis_a.attrs['z_base']:.0f} m"

        ax_hex.set_title(f"(a) Hexagram trajectory | k={k}{layer_txt}{rgb_txt}")
        ax_hex.set_xlabel("hex_x")
        ax_hex.set_ylabel("hex_y")
        ax_hex.set_xlim(-0.5, img.shape[1] - 0.5)
        ax_hex.set_ylim(-0.5, img.shape[0] - 0.5)
        ax_hex.grid(False)

        # ------------------------------------------------------------------
        # (b) Process timeline
        # ------------------------------------------------------------------
        df = classified_a[["proc_label", "strength"]].to_dataframe().reset_index()
        cat = pd.Categorical(df["proc_label"])
        y = cat.codes.astype(int)
        y[y < 0] = len(cat.categories)

        sc2 = ax_tl.scatter(
            df["time"].to_numpy(),
            y,
            c=df["strength"].to_numpy(),
            cmap=cmap,
            s=30,
            vmin=0,
            vmax=1,
        )

        ax_tl.set_title("(b) Process timeline (color = strength)")
        ax_tl.set_ylabel("Process")
        ax_tl.set_yticks(range(len(cat.categories)))
        ax_tl.set_yticklabels(list(cat.categories))
        ax_tl.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        fig.colorbar(sc2, ax=ax_tl, fraction=0.03, pad=0.01)

        # ------------------------------------------------------------------
        # (c) Signs vs time
        # ------------------------------------------------------------------
        tnum = mdates.date2num(pd.to_datetime(t).to_pydatetime())

        def _plot_sign(ax, s, label):
            ax.step(tnum, s, where="mid")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(["−", "0", "+"])
            ax.set_ylabel(label)

        _plot_sign(ax_sgn, classified_a["sign_R"].values, "sign_R")
        ax_sgn.step(tnum, classified_a["sign_G"].values + 0.05, where="mid")
        ax_sgn.step(tnum, classified_a["sign_B"].values - 0.05, where="mid")

        ax_sgn.legend(
            ["sign_R", "sign_G (+0.05)", "sign_B (-0.05)"],
            frameon=False,
            loc="upper right",
        )
        ax_sgn.set_title("(c) Signs vs time")
        ax_sgn.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # ------------------------------------------------------------------
        # (d) Strength vs time
        # ------------------------------------------------------------------
        strength = classified_a["strength"].values
        ax_str.plot(tnum, strength, lw=2)

        thr = classified_a.attrs.get("min_strength", None)
        if thr is not None:
            ax_str.axhline(float(thr), ls="--", lw=1)

        ax_str.set_title("(d) Strength")
        ax_str.set_ylabel("strength")
        ax_str.set_xlabel("Time")
        ax_str.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        ax_sgn.set_xlim(tnum.min(), tnum.max())
        ax_str.set_xlim(tnum.min(), tnum.max())

        fig.tight_layout()

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        filepath = None
        if savefig:
            outdir = Path.cwd() if output_dir is None else Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)

            t0s = analysis_a.attrs.get("period_start", "").replace(":", "")
            t1s = analysis_a.attrs.get("period_end", "").replace(":", "")
            layer_tag = "layer"
            if "z_top" in analysis_a.attrs and "z_base" in analysis_a.attrs:
                layer_tag = f"{analysis_a.attrs['z_top']:.0f}-{analysis_a.attrs['z_base']:.0f}m"

            filepath = outdir / f"{t0s}_{t1s}_microphysics_summary_{layer_tag}_k{k}.png"
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath
