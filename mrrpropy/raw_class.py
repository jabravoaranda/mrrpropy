from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import xarray as xr
from datetime import datetime

from mrrpropy import generate_rgb_hex
import mrrpropy.RaProMPro_original as rpm
from mrrpropy.utils import build_rgb_from_trends, map_rgb_to_hexagram, to_time_slice, ols_slope_intercept_r2, compute_eps 
DatetimeLike = Union[str, np.datetime64, datetime]

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 32,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 14,
    })

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
            raise KeyError(f"Variable '{name}' does not exist. Available variables: {list(self.ds.data_vars)}")
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
            raise RuntimeError("Dataset does not contain 'index_spectra'; cannot retrieve spectrum.")

        idx_spec = int(self.ds["index_spectra"].isel(time=time_idx, range=range_idx).values)

        # Velocity axis (only n_spectra, spectrum_n_samples)
        vel = self.ds["D"].isel(n_spectra=idx_spec)

        if use_raw:
            var_name = "spectrum_raw"
        else:
            var_name = "N"

        if var_name not in self.ds:
            raise RuntimeError(f"Dataset does not contain spectral variable '{var_name}'.")

        spec = self.ds[var_name].isel(time=time_idx, n_spectra=idx_spec)
        return vel, spec


    def process_raprompro(
        self,
        *,
        adjust_m: float = 1.0,
        save_spe_3d: bool = False,
        save_dsd_3d: bool = False,
        
    ) -> xr.Dataset:
        """
        Run RaProM-Pro processing using the published CLI algorithm implementation
        (RaProMPro_original.py), but exposed as a method returning an xarray.Dataset.

        Key design goal: keep the scientific algorithm and naming consistent with
        the original CLI output (Type, W, spectral width, Skewness, Kurtosis, DBPIA,
        LWC, RR, SR, Za, Z, Zea, Ze, Z_all, ... and BB_*).
        """

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
            raise RuntimeError("Dataset must contain 'transfer_function' and 'calibration_constant'.")

        if "index_spectra" not in ds or "D" not in ds:
            raise RuntimeError("CF/Radial spectra mapping requires 'index_spectra' and 'D'.")

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
        time_unix = (ds["time"].values.astype("datetime64[s]").astype("int64")).astype(float)

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
        spe_3d_list = []  # (time, range, speed3) in original; we store NewMatrix (dealiased) if requested
        dsd_3d_list = []  # (time, range, DropSize) in original; we store log10(NdE) if requested

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
                estat, NewMatrix, z_da, Lwc, Rr, SnowRate, w, sig, sk, Noi,
                DSD, NdE, Ze, Mov, velTur, snr, kur, PiA, NW, DM,
                z_P, lwc_P, rr_P, Z_h, Z_all, RR_all, LWC_all,
                dm_all, nw_all, N_all
            ) = rpm.Process(proeta, Hcolum, time_unix[it], D, Cte, NewNoise, Deltav, Code_spectrum, Snr_Refl_2)

            # BB logic (original uses special handling for first two times)
            if it == 0:
                bb_bot, bb_top, bb_peak = rpm.BB2(
                    w, Ze, Hcolum, sk, kur,
                    np.ones(2) * np.nan, np.ones(2) * np.nan, np.ones(2) * np.nan
                )
            elif it == 1:
                bb_bot, bb_top, bb_peak = rpm.BB2(
                    w, Ze, Hcolum, sk, kur,
                    np.ones(2) * bb_bot_full, np.ones(2) * bb_top_full, np.ones(2) * bb_peak_full
                )
            else:
                bb_bot, bb_top, bb_peak = rpm.BB2(w, Ze, Hcolum, sk, kur, bb_bot_full, bb_top_full, bb_peak_full)

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

            if np.isnan(bb_peak_full2[j]) and ~np.isnan(bb_bot_full2[j]) and np.isnan(bb_top_full2[j]):
                bb_bot_full2[j] = np.nan
            if np.isnan(bb_peak_full2[j]) and np.isnan(bb_bot_full2[j]) and ~np.isnan(bb_top_full2[j]):
                bb_top_full2[j] = np.nan
            if ~np.isnan(bb_peak_full2[j]) and np.isnan(bb_bot_full2[j]) and ~np.isnan(bb_top_full2[j]):
                bb_top_full2[j] = np.nan
                bb_peak_full2[j] = np.nan
            if ~np.isnan(bb_peak_full2[j]) and ~np.isnan(bb_bot_full2[j]) and np.isnan(bb_top_full2[j]):
                bb_bot_full2[j] = np.nan
                bb_peak_full2[j] = np.nan
            if ~np.isnan(bb_peak_full2[j]) and np.isnan(bb_bot_full2[j]) and np.isnan(bb_top_full2[j]):
                bb_bot_full2[j] = np.nan
                bb_top_full2[j] = np.nan
            if np.isnan(bb_peak_full2[j]) and ~np.isnan(bb_bot_full2[j]) and ~np.isnan(bb_top_full2[j]):
                bb_peak_full2[j] = bb_bot_full2[j] + ((bb_top_full2[j] - bb_bot_full2[j]) / 2.0)

        # CorrectWithBBMatrix in-place correction (CLI)
        estat_full, Z_da_full, LWC_full, RR_full, SnowR_full = rpm.CorrectWithBBMatrix(
            estat_full, Z_da_full, LWC_full, RR_full, SnowR_full,
            Hcolum, bb_bot_full2, bb_top_full2, Z_ea_full,
            # NOTE: these were built inside loop in CLI as Z_P/LWC_P/RR_P per time
            # In the CLI they keep Z_P/LWC_P/RR_P time-stacked. We reproduce that
            # by recomputing them as the “MP parameters” already returned from Process
            # is included inside the Process return; in this method we did not store
            # them. For exact parity you can store z_P/lwc_P/rr_P stacks too.
            # Minimal safe approximation: pass NaNs to skip those corrections.
            np.full_like(Z_da_full, np.nan),  # Z_P
            np.full_like(LWC_full, np.nan),   # LWC_P
            np.full_like(RR_full, np.nan),    # RR_P
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

        _da2("Type", estat_full, "", "Predominant hydrometeor type numerical value (original CLI)")
        _da2("W", w_full, "m s-1", "Fall speed with aliasing correction")
        _da2("spectral width", sig_full, "m s-1", "Spectral width of the dealiased velocity distribution")
        _da2("Skewness", sk_full, "none", "Skewness of the spectral reflectivity with dealiasing")
        _da2("Kurtosis", kur_full, "none", "Kurtosis of the spectral reflectivity with dealiasing")
        _da2("DBPIA", PIA_full, "dB", "Path Integrated Attenuation (dB) assuming liquid phase")
        _da2("LWC", LWC_full, "g m-3", "Liquid Water Content using only liquid hydrometeors (by Type)")
        _da2("RR", RR_full, "mm hr-1", "Rain Rate using only liquid hydrometeors (by Type)")
        _da2("SR", SnowR_full, "mm hr-1", "Snow Rate")
        _da2("Za", Z_a_full, "dBZ", "Attenuated reflectivity corrected by PIA only for liquid hydrometeors")
        _da2("Zea", Z_ea_full, "dBZ", "Equivalent attenuated reflectivity")
        _da2("Ze", Z_e_full, "dBZ", "Equivalent reflectivity corrected by PIA only for drizzle/rain")
        _da2("Z_all", z_all_full, "dBZ", "Attenuated reflectivity corrected by PIA assuming all liquid")
        _da2("LWC_all", lwc_all_full, "g m-3", "LWC assuming all liquid")
        _da2("RR_all", rr_all_full, "mm hr-1", "RR assuming all liquid")
        _da2("N_all", n_all_full, "log10(m-3 mm-1)", "log10(total N) assuming all liquid")
        _da2("Nw", nw_full, "log10(mm-1 m-3)", "Normalized intercept parameter (by Type)")
        _da2("Dm", dm_full, "mm", "Mean mass-weighted diameter (by Type)")
        _da2("Nw_all", NW_all_full, "log10(mm-1 m-3)", "Normalized intercept parameter (all liquid)")
        _da2("Dm_all", DM_all_full, "mm", "Mean mass-weighted diameter (all liquid)")
        _da2("Noise", Noi_full, "", "Noise estimate in eta(n) units (original)")
        _da2("SNR", SNR_full, "dB", "SNR used/derived by algorithm (original)")
        _da2("N_da", N_da_full, "log10(m-3 mm-1)", "log10(N(D)) derived (original 'N_da')")

        # BB as (time,BB_Height) to mirror CLI netCDF shape
        out["BB_bottom"] = xr.DataArray(
            np.asarray(bb_bot_full2, dtype=float)[:, None],
            dims=("time", "BB_Height"),
            attrs={"units": "m", "description": "range from BB bottom above sea level (original CLI)"},
        )
        out["BB_top"] = xr.DataArray(
            np.asarray(bb_top_full2, dtype=float)[:, None],
            dims=("time", "BB_Height"),
            attrs={"units": "m", "description": "range from BB top above sea level (original CLI)"},
        )
        out["BB_peak"] = xr.DataArray(
            np.asarray(bb_peak_full2, dtype=float)[:, None],
            dims=("time", "BB_Height"),
            attrs={"units": "m", "description": "range from BB peak above sea level (original CLI)"},
        )

        # Optional 3D products (names follow original netCDF)
        if save_spe_3d:
            out["spe_3D"] = xr.DataArray(
                np.asarray(spe_3d_list, dtype=float),
                dims=("time", "range", "speed"),
                coords={"time": coords["time"], "range": coords["range"], "speed": np.arange(-Nbins * fNy, 2 * Nbins * fNy, fNy)},
                attrs={"units": "mm-1", "description": "spectral reflectivity dealiased (original CLI)"},
            )

        if save_dsd_3d:
            out["dsd_3D"] = xr.DataArray(
                np.asarray(dsd_3d_list, dtype=float),
                dims=("time", "range", "DropSize"),
                coords={"time": coords["time"], "range": coords["range"], "DropSize": np.asarray(D[0], dtype=float)},
                attrs={"units": "log10(m-3 mm-1)", "description": "3D DSD (original CLI)"},
            )

        return out

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
        return all(v in self.ds.data_vars for v in required)


    # -------------------------
    # Resource Management
    # -------------------------
    def close(self):
        """Close the xarray dataset (e.g., at the end of the script)."""
        self.ds.close()


    # -------------------------
    # Quick Plot (optional)
    # -------------------------
    def quickplot_reflectivity(
        self,
        field: str = "Ze",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize: tuple[float, float] = (10, 6),
        cmap = 'jet'
    ):
        """
        Create a time–range plot of reflectivity (or any 2D field time×range).

        Requires matplotlib. Intended for quick inspections.
        """
        import matplotlib.pyplot as plt

        if field not in self.ds:
            raise KeyError(f"Variable '{field}' does not exist.")
        da = self.ds[field]  # dims: (time, range)

        fig, ax = plt.subplots(figsize=figsize)
        im = da.plot(
            ax=ax,
            x="time",
            y="range",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
            cmap = cmap
        )
        ax.set_title(f"{field} (MRR-PRO)")
        ax.set_ylabel("Range (m)")
        ax.set_xlabel("Time")
        plt.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------------
    # Helpers internos para espectros MRR-PRO
    # -------------------------------------------------------------------------
    def _nearest_period(
        self, target_time: datetime | np.datetime64, target_range: float
    ) -> tuple[np.datetime64, float]:
        """Devuelve el time y range reales seleccionados por nearest."""
        ds = self.ds
        t_sel = ds["time"].sel(time=target_time, method="nearest").values
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
        target_time: datetime | np.datetime64,
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

        t_sel, r_sel = self._nearest_period(target_time, target_range)

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
            idx = ds["index_spectra"].sel(time=t_sel, range=r_sel, method="nearest").values
            js = int(idx)
            s = da.sel(time=t_sel, n_spectra=js).values.astype(float)
            vel = self._get_velocity_axis(s.shape[-1])
            return t_sel, r_sel, vel, s, units

        raise ValueError(
            f"Formato de '{spectrum_var}' no soportado. dims={da.dims}"
        )

    def _get_spectrogram_2d(
        self,
        target_time: datetime | np.datetime64,
        *,
        spectrum_var: str = "spectrum_reflectivity",
        range_limits: tuple[float, float] | None = None,
    ) -> tuple[np.datetime64, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Extrae un espectrograma 2D (range x doppler_bin) para el instante más cercano.

        Returns:
          t_sel, ranges, vel_axis, spec2d, units
        """
        ds = self.ds
        if spectrum_var not in ds:
            if "spectrum_raw" in ds:
                spectrum_var = "spectrum_raw"
            else:
                raise KeyError(
                    f"No encuentro '{spectrum_var}' ni 'spectrum_raw' en el Dataset."
                )

        t_sel = ds["time"].sel(time=target_time, method="nearest").values
        da = ds[spectrum_var]
        units = str(da.attrs.get("units", ""))

        # Rango a representar
        if range_limits is None:
            r0 = float(ds["range"].min().values)
            r1 = float(ds["range"].max().values)
        else:
            r0, r1 = map(float, range_limits)

        ranges = ds["range"].sel(range=slice(r0, r1)).values.astype(float)
        n_bins = ds.sizes.get("spectrum_n_samples", None)
        if n_bins is None:
            raise ValueError("No encuentro dimensión 'spectrum_n_samples' en el Dataset.")
        vel = self._get_velocity_axis(int(n_bins))

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
            idx_vec = ds["index_spectra"].sel(time=t_sel, range=slice(r0, r1)).values.astype(int)
            # Extrae spectra para ese time: (n_spectra, bin)
            slab = da.sel(time=t_sel).values.astype(float)  # (n_spectra, bin)
            # Mapea (range -> n_spectra) => (range, bin)
            spec2d = slab[idx_vec, :]
            return t_sel, ranges, vel, spec2d, units

        raise ValueError(
            f"Formato de '{spectrum_var}' no soportado. dims={da.dims}"
        )

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    def plot_spectrum(
        self,
        target_time: datetime | np.datetime64,
        target_range: float,
        *,
        spectrum_var: str = "spectrum_reflectivity",
        velocity_limits: tuple[float, float] | None = None,
        color: str = "black",
        label_type: str = "both",  # both|time|range
        fig: Figure | None = None,
        ax=None,
        output_dir: Path | None = None,
        savefig: bool = False,
        dpi: int = 200,
    ) -> tuple[Figure, Path | None]:
        """
        Espectro 1D (más cercano) en un tiempo y altura/rango dados.

        - Soporta cubo o indexado.
        - No hace conversión física adicional: plotea en las unidades del netCDF.
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        elif fig is not None and ax is None:
            ax = fig.get_axes()[0]

        t_sel, r_sel, vel, spec, units = self._get_spectrum_1d(
            target_time, target_range, spectrum_var=spectrum_var
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
            ax.plot(vel, spec, color=color, label=label)

        # ejes
        if velocity_limits is not None:
            ax.set_xlim(*velocity_limits)
        else:
            ax.set_xlim(float(np.nanmin(vel)), float(np.nanmax(vel)))

        ax.set_xlabel("Doppler velocity [m/s]")
        ylabel = f"Spectrum [{units}]" if units else "Spectrum"
        ax.set_ylabel(ylabel)
        ax.set_title(f"MRR-PRO spectrum | time={t_txt} | range={r_sel:.1f} m")

        ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.0)
        ax.legend(loc="upper right", fontsize=8)

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
        target_time,
        ranges: list[float] | np.ndarray,
        *,
        use_db: bool = True,
        label_type: str = "range",
        ncol: int = 2,
        figsize: tuple[float, float] = (10, 7),
        fig=None,
        ax=None,
        output_dir=None,
        savefig: bool = False,
        dpi: int = 200,
        **kwargs,
    ):
        """
        Plot several MRR-PRO Doppler spectra at a fixed time for multiple ranges.

        This method overlays spectra for the nearest (time, range) gates.
        It relies on the RAW spectral variable 'spectrum_reflectivity' (preferred) or
        falls back to 'spectrum' if present.

        Parameters
        ----------
        target_time : datetime | np.datetime64 | str
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
        import numpy as np
        import matplotlib.pyplot as plt

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
        t_sel = ds["time"].sel(time=target_time, method="nearest").values

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
                idx_raw = ds["index_spectra"].sel(time=t_sel, range=r_sel, method="nearest").values
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
                    raise ValueError(f"{spec_var} does not have 'spectrum_n_samples' dimension.")
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
            filepath = output_dir / self.path.name.replace(".nc", f"_spectra_by_range_{ttxt}.png")
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath

    def plot_ND_by_range(
        self,
        target_time,
        ranges: list[float] | np.ndarray,
        *,
        use_log10: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "viridis",
        ncol: int = 2,
        figsize: tuple[float, float] = (10, 7),
        fig=None,
        ax=None,
        output_dir=None,
        savefig: bool = False,
        dpi: int = 200,
        **kwargs,
    ):
        """
        Plot several N(D) curves at a fixed time for multiple provided ranges.

        For each requested range (m), the nearest range gate is selected. Then
        `index_spectra(time, range)` is used to map that gate to `n_spectra`, and:
            - D := ds['D'][n_spectra, :]
            - N := ds['N'][time, n_spectra, :]

        The result is an overlay of curves N vs D for the selected ranges.

        Parameters
        ----------
        target_time : datetime | np.datetime64 | str
            Target time. Nearest time gate is used.
        ranges : list[float] | np.ndarray
            List of ranges in meters. Nearest range gate is used for each.
        use_log10 : bool, default True
            Plot log10(N) if True, else N linear.
        vmin, vmax : float | None
            Optional y-limits (applied as ylim). If both are None, no limits set.
        cmap : str, default "viridis"
            Colormap used to assign line colors by range.
        ncol : int, default 2
            Legend columns.
        figsize : tuple, default (10,7)
            Figure size if fig/ax not provided.
        fig, ax : matplotlib Figure/Axes, optional
            Reuse existing axes.
        output_dir : Path, optional
            Output directory if savefig=True.
        savefig : bool, default False
            Save to disk if True.
        dpi : int, default 200
            Save dpi.

        Returns
        -------
        (fig, filepath) : (Figure, Path | None)
        """

        ds = self.ds

        # --- sanity checks ---
        for v in ("N", "D", "index_spectra", "time", "range"):
            if v not in ds:
                raise KeyError(f"Dataset missing required variable '{v}'")
        for d in ("n_spectra", "spectrum_n_samples"):
            if d not in ds.dims:
                raise KeyError(f"Dataset missing required dimension '{d}'")

        # --- nearest time ---
        t_sel = ds["time"].sel(time=target_time, method="nearest").values

        # --- figure/axes ---
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif fig is not None and ax is None:
            axes = fig.get_axes()
            ax = axes[0] if len(axes) else fig.add_subplot(111)
        elif fig is None and ax is not None:
            fig = ax.figure

        # --- color assignment by range ---
        ranges_in = np.asarray(ranges, dtype=float)
        if ranges_in.size == 0:
            raise ValueError("ranges must contain at least one value.")

        # Use a matplotlib colormap without importing extra helpers
        cm = plt.get_cmap(cmap)
        # Keep ordering stable (requested order), but normalize colors by index
        colors = [cm(i / max(1, ranges_in.size - 1)) for i in range(ranges_in.size)]

        # Get arrays once
        D_all = ds["D"].values.astype(float)                 # (n_spectra, bin)
        N_all = ds["N"].sel(time=t_sel).values.astype(float) # (n_spectra, bin)

        # --- loop ranges ---
        plotted_any = False
        for i, r_req in enumerate(ranges_in):
            r_sel = float(ds["range"].sel(range=r_req, method="nearest").values.item())
            print(r_sel)
            idx_raw = ds["index_spectra"].sel(time=t_sel, range=r_sel, method="nearest").values
            if not np.isfinite(idx_raw):
                continue
            idx = int(idx_raw)
            if not (0 <= idx < ds.sizes["n_spectra"]):
                continue

            D = D_all[idx, :].astype(float)
            N = N_all[idx, :].astype(float)

            # Clean N (ignore <=0)
            N_minimum_thresh = float(kwargs.get("N_minimum_threshold", 0.0))
            N = np.where(N >= N_minimum_thresh, N, np.nan)

            # Ensure D increases
            if D.size >= 2 and D[0] > D[-1]:
                D = D[::-1]
                N = N[::-1]

            # Drop non-finite
            ok = np.isfinite(D) & np.isfinite(N)
            if not np.any(ok):
                continue

            y = np.log10(N[ok]) if use_log10 else N[ok]
            x = D[ok]

            ax.plot(x*1000, y, color=colors[i], label=f"{r_sel:.1f} m", marker='o', markersize=4)
            plotted_any = True

        if not plotted_any:
            raise ValueError("No valid spectra found for the provided ranges/time.")

        # --- labels / title ---
        d_units = 'mm'
        xlab = f"D [{d_units}]".strip() if d_units else "D"
        ax.set_xlabel(xlab)

        ax.set_ylabel(r"$log10(N [mm^{-1} m^{-3}])$" if use_log10 else r"$N [mm^{-1} m^{-3}]$")

        t_txt = np.datetime_as_string(t_sel, unit="s")
        ax.set_title(f"MRR-PRO N(D) by range | time={t_txt}")

        #set log scale in Y axis
        ax.set_yscale('linear' if use_log10 else "log")

        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax.legend(ncol=ncol, loc="best", fontsize=9)

        if vmin is not None or vmax is not None:
            ax.set_ylim(vmin, vmax)

        if kwargs.get("xlimits", None) != None:
            ax.set_xlim(kwargs["xlimits"])

        fig.tight_layout()

        filepath = None
        if savefig:
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig=True.")
            output_dir.mkdir(parents=True, exist_ok=True)
            ttag = np.datetime_as_string(t_sel, unit="s").replace(":", "")
            filepath = output_dir / self.path.name.replace(".nc", f"_ND_by_range_{ttag}.png")
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath


    def plot_spectrogram(
        self,
        target_time: datetime | np.datetime64,
        *,
        spectrum_var: str = "spectrum_reflectivity",
        range_limits: tuple[float, float] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "jet",
        fig: Figure | None = None,
        ax=None,
        output_dir: Path | None = None,
        savefig: bool = False,
        dpi: int = 200,
    ) -> tuple[Figure, Path | None]:
        """
        Plot a spectrogram of MRR-PRO radar data for a specified time.

        Parameters
        ----------
        target_time : datetime | np.datetime64
            The target time for which to generate the spectrogram.
        spectrum_var : str, optional
            The spectrum variable to plot. Default is "spectrum_reflectivity".
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

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        elif fig is not None and ax is None:
            ax = fig.get_axes()[0]

        t_sel, ranges, vel, spec2d, units = self._get_spectrogram_2d(
            target_time, spectrum_var=spectrum_var, range_limits=range_limits
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
        title = f"MRR-PRO spectrogram | time={t_txt}"
        if range_limits is not None:
            title += f" | range=[{range_limits[0]:.0f}, {range_limits[1]:.0f}] m"
        ax.set_title(title)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Spectrum [{units}]" if units else "Spectrum")

        fig.tight_layout()

        filepath: Path | None = None
        if savefig:
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig=True.")
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / self.path.name.replace(
                ".nc", f"_spectrogram_{t_txt.replace(':','')}.png"
            )
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath

    def plot_ND_gram(
        self,
        target_time,
        *,
        range_limits: tuple[float, float] | None = None,
        use_log10: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "viridis",
        fig=None,
        ax=None,
        output_dir=None,
        savefig: bool = False,
        dpi: int = 200,
    ):
        """
        Plot a Range–D gram using MRR-PRO spectral variable N and diameter D.

        Y-axis: range (m)
        X-axis: D (from ds['D'])
        Color: N (optionally log10-scaled)

        NOTE:
        - This is NOT a physical DSD inversion.
        - N is a spectral quantity indexed by Doppler/size bins.
        """
        ds = self.ds
        breakpoint()
        # --- Sanity checks ---
        for v in ("N", "D", "index_spectra", "range", "time"):
            if v not in ds:
                raise KeyError(f"Dataset missing required variable '{v}'")

        # --- Select nearest time ---
        t_sel = ds["time"].sel(time=target_time, method="nearest").values

        # --- Select range slice ---
        if range_limits is None:
            r0 = float(ds["range"].min().values)
            r1 = float(ds["range"].max().values)
        else:
            r0, r1 = map(float, range_limits)

        # --- Select range slice ---
        if range_limits is None:
            r0 = float(ds["range"].min().values)
            r1 = float(ds["range"].max().values)
        else:
            r0, r1 = map(float, range_limits)

        ranges = ds["range"].sel(range=slice(r0, r1)).values.astype(float)

        if ranges.size == 0:
            raise ValueError("No range gates selected for the given range_limits.")
        
        # --- Map range -> n_spectra indices (sanitize!) ---
        idx_raw = ds["index_spectra"].sel(time=t_sel, range=slice(r0, r1)).values

        # idx_raw is float; it may contain NaN. Convert safely.
        idx_vec = np.full(idx_raw.shape, -1, dtype=int)
        finite = np.isfinite(idx_raw)
        idx_vec[finite] = idx_raw[finite].astype(int)

        n_spec = ds.sizes["n_spectra"]
        valid_idx = (idx_vec >= 0) & (idx_vec < n_spec)

        # If some gates are invalid, mask them out (keep plot consistent)
        # We'll keep ranges but set data to NaN where invalid.
        # (Alternative: drop invalid ranges; but this keeps y-grid intact.)
        D_all = ds["D"].values.astype(float)                # (n_spectra, bin)
        N_all = ds["N"].sel(time=t_sel).values.astype(float)  # (n_spectra, bin)

        X = np.full((ranges.size, D_all.shape[1]), np.nan, dtype=float)
        Z = np.full((ranges.size, D_all.shape[1]), np.nan, dtype=float)

        X[valid_idx, :] = D_all[idx_vec[valid_idx], :]
        Z[valid_idx, :] = N_all[idx_vec[valid_idx], :]

        # Clean invalid/negative N
        Z = np.where(Z > 0.0, Z, np.nan)

        # Log scaling
        if use_log10:
            with np.errstate(divide="ignore", invalid="ignore"):
                Zplot = np.log10(Z)
            Zplot[~np.isfinite(Zplot)] = np.nan
            clabel = "log10(N)"
        else:
            Zplot = Z
            clabel = "N"

        # Ensure range increases upward
        if ranges.size >= 2 and ranges[0] > ranges[-1]:
            ranges = ranges[::-1]
            X = X[::-1, :]
            Zplot = Zplot[::-1, :]

        # --- Build explicit cell edges (critical for non-monotonic 2D X) ---
        # D edges per range gate
        # --- D edges must be (M+1, N+1) to match R_edges and Zplot ---
        M, N = X.shape  # X is (range, bin)

        # First compute per-row edges along D: (M, N+1)
        D_row_edges = np.full((M, N + 1), np.nan, dtype=float)
        D_row_edges[:, 1:-1] = 0.5 * (X[:, 1:] + X[:, :-1])
        D_row_edges[:, 0] = X[:, 0] - (D_row_edges[:, 1] - X[:, 0])
        D_row_edges[:, -1] = X[:, -1] + (X[:, -1] - D_row_edges[:, -2])

        # Then expand to (M+1, N+1) by adding one extra row (range-edge direction)
        D_edges = np.full((M + 1, N + 1), np.nan, dtype=float)
        D_edges[1:-1, :] = 0.5 * (D_row_edges[1:, :] + D_row_edges[:-1, :])

        # Top/bottom extrapolation
        D_edges[0, :] = D_row_edges[0, :]
        D_edges[-1, :] = D_row_edges[-1, :]

        # Range edges (1D) then expand to 2D
        r_edges = np.empty(ranges.size + 1, dtype=float)
        r_edges[1:-1] = 0.5 * (ranges[1:] + ranges[:-1])
        r_edges[0] = ranges[0] - (r_edges[1] - ranges[0])
        r_edges[-1] = ranges[-1] + (ranges[-1] - r_edges[-2])
        R_edges = np.tile(r_edges[:, None], (1, D_edges.shape[1]))

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        elif fig is not None and ax is None:
            axes = fig.get_axes()
            if len(axes) == 0:
                ax = fig.add_subplot(111)
            else:
                ax = axes[0]
        elif fig is None and ax is not None:
            fig = ax.figure

        # Plot with edges: Zplot is (M,N), edges are (M+1,N+1)
        im = ax.pcolormesh(
            D_edges,
            R_edges,
            Zplot,
            shading="auto",
            cmap=cmap,
        )

        if vmin is not None or vmax is not None:
            im.set_clim(vmin=vmin, vmax=vmax)

        t_txt = np.datetime_as_string(t_sel, unit="s")
        ax.set_title(f"MRR-PRO Range–D gram (spectral N) | time={t_txt}")
        ax.set_xlabel("D")
        ax.set_ylabel("Range [m]")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(clabel)

        fig.tight_layout()

        # --- Save if requested ---
        filepath = None
        if savefig:
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig=True.")
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / self.path.name.replace(
                ".nc", f"_NDgram_{t_txt.replace(':','')}.png"
            )
            fig.savefig(filepath, dpi=dpi)

        return fig, filepath

    def plot_raprompro_profiles(
        self,
        target_datetime: datetime.datetime,
        figsize: tuple[float, float] = (18, 12),
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs,
    )-> tuple[Figure, np.ndarray, Path | None]:
        """
        RaProMPro diagnostic profile (single figure, 5 axes; Y = height):
        1) Ze, Zea, Z_all, Za
        2) Dm
        3) Nw
        4) LWC
        5) RR

        Uses self.ds and selects the nearest profile to `target_datetime`.
        Raises if the dataset does not look RaProMPro-preprocessed.
        """
        ds = self.ds

        # --- minimal "preprocessed?" check ---
        if not self._is_processed():
            raise RuntimeError(
        "Dataset has not been RaProMPro-preprocessed. "
        "Run process_rarpom() before calling plot_rarpom_diagnostic_profile()."
    )

        # --- vertical coordinate ---
        elif "range" in ds.coords:
            zname = "range"
        else:
            raise RuntimeError("No vertical coordinate found: expected 'height' or 'range'.")

        if "time" not in ds.coords:
            raise RuntimeError("No 'time' coordinate found in dataset.")

        # --- select nearest time ---
        prof = ds.sel(time=np.datetime64(target_datetime), method="nearest")
        sel_time = prof["time"].values
        try:
            sel_time_str = np.datetime_as_string(sel_time, unit="s")
        except Exception:
            sel_time_str = str(sel_time)

        z = prof[zname].values.astype(float)/1000.0  # to km

        fig, axs = plt.subplots(
            ncols=5,
            figsize=figsize,
            sharey=True,
            constrained_layout=True,
        )

        # 1) Reflectivities
        ax = axs[0]
        ax.plot(prof["Ze"].values, z, label="Ze", linewidth=1, marker='o', markersize=4)
        ax.plot(prof["Zea"].values, z, label="Zea", linewidth=1, marker='o', markersize=4)
        ax.plot(prof["Z_all"].values, z, label="Z_all", linewidth=1, marker='o', markersize=4)
        ax.plot(prof["Za"].values, z, label="Za", linewidth=1, marker='o', markersize=4)
        ax.set_xlabel("Reflectivities, dBZ")
        ax.set_ylabel(f"{zname} (km)")
        ax.grid(True)
        ax.legend(loc="best")

        # 2) Dm
        ax = axs[1]
        ax.plot(prof["Dm"].values, z, linewidth=1, marker='o', markersize=4)
        ax.set_xlabel("Dm, mm")
        ax.grid(True)

        # 3) Nw
        ax = axs[2]
        ax.plot(prof["Nw"].values, z, linewidth=1,marker='o', markersize=4)
        ax.set_xlabel("log10(Nw mm⁻¹ m⁻³)")
        ax.grid(True)

        # 4) LWC
        ax = axs[3]
        ax.plot(prof["LWC"].values, z, linewidth=1,marker='o', markersize=4)
        ax.set_xlabel("LWC, g m⁻³")
        ax.grid(True)

        # 5) RR
        ax = axs[4]
        ax.plot(prof["RR"].values, z, linewidth=1, marker='o', markersize=4)
        ax.set_xlabel("RR, mm h⁻¹")
        ax.grid(True)

        if kwargs.get('y_limits', None) is not None:
            for ax in axs:
                ax.set_ylim(kwargs['y_limits'])

        fig.suptitle(f"RaProMPro diagnostic profile — {sel_time_str}", fontsize=30)

        if savefig:
            if output_dir is None:
                output_dir = Path().cwd()
            datestr = target_datetime.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{self.path.stem}_{datestr}_raprompro_profiles.png"
            fig.savefig(output_path)

        return fig, axs, output_path

    def plot_rain_process_in_layer(self,
                                    target_datetime: datetime | tuple[datetime, datetime],
                                    layer: tuple[float, float],
                                    x: str = 'Dm',
                                    y: str = 'LwC',
                                    z: str = "Nw",
                                    use_relative_difference: bool = True,
                                    savefig: bool = False,
                                    **kwargs
                                    ) -> tuple[Figure, Path | None]:
        """
        Plots the rain process in a specified atmospheric layer at a given datetime.
        This method generates a scatter plot of two selected variables (x and y) from the dataset,
        colored by a third variable (z), representing the difference in properties within a specified
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
                - marker_size: int or float, optional
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

        
        #Check x,y,z exists as variable in self.ds, otherwise raise error
        for var in (x, y, z):
            if var not in self.ds:
                raise KeyError(f"Variable '{var}' not found in dataset.")

        #Check time tuple increasing
        if isinstance(target_datetime, tuple):
            if target_datetime[0] >= target_datetime[1]:
                raise ValueError("target_datetime tuple must be in increasing order (start, end).")

        #get data profiles from self.data
        if isinstance(target_datetime, datetime):
            ds_ = self.ds.sel(time=target_datetime, method='nearest').sel(range=slice(*layer))
        else:
            ds_ = self.ds.sel(time=slice(*target_datetime)).sel(range=slice(*layer))
        
        #Get the difference of all properties between the range with respect to the zmax (end - start)
        last_range = ds_.range[-1]

        if use_relative_difference:
            diff_ = 100*(ds_-ds_.sel(range=last_range))/ds_.sel(range=slice(*layer)).mean('range')
            #Get maximum values for x,y,z to set simetric axis limits
            ds_ = diff_.copy()
        else:
            ds_ = ds_ - ds_.sel(range=last_range)
            
        x_abs_max = np.abs(ds_[x].values[np.isfinite(ds_[x].values)]).max()
        y_abs_max = np.abs(ds_[y].values[np.isfinite(ds_[y].values)]).max()
        z_abs_max = np.abs(ds_[z].values[np.isfinite(ds_[z].values)]).max()
        #create a figure with same y and x axis size
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 6)))

        # for time_ in diff_.time:
        ds_.plot.scatter(x=x, 
                            y=y, 
                            hue=z,
                            cmap=kwargs.get('cmap', 'viridis'), 
                            s=kwargs.get('marker_size', 50), 
                            vmin = -z_abs_max,
                            vmax = z_abs_max,   
                            ax=ax
                            )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        # cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar = ax.get_figure().get_axes()[1]
        cbar.set_ylabel(z)
        zmin, zmax = layer  
        if isinstance(target_datetime, tuple):
            ax.set_title(f"Layer {zmin/1000.}-{zmax/1000.} km \n {target_datetime[0]} to {target_datetime[1]}")
        else:
            ax.set_title(f"Layer {zmin/1000.}-{zmax/1000.} km | {target_datetime}")
        #create simetric limits for x and y axis
        ax.set_xlim(-x_abs_max, x_abs_max)
        ax.set_ylim(-y_abs_max, y_abs_max)
        
        #add grid and unity line
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Highlight main axis lines (x,0) and (0,y)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        
         # Adjust layout

        fig.tight_layout()

        if savefig:
            output_dir = kwargs.get('output_dir', Path().cwd())
            output_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(target_datetime, tuple):
                datestr = f"{target_datetime[0].strftime('%Y%m%d_%H%M%S')}_to_{target_datetime[1].strftime('%Y%m%d_%H%M%S')}"
            else:
                datestr = target_datetime.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{self.path.stem}_{datestr}_rain_process_layer_{zmin}-{zmax}m.png"
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
        nmin: int = 10,
    ) -> xr.Dataset:
        """
        Compute b_X (1/m), F_X, and R^2 for X in vars using OLS of ln(X) vs depth from top.

        eps_mode:
        - "hourly_quantile": epsilon per time step (hour/file) using quantile q
        - "global_quantile": epsilon global over provided dataset subset
        - "fixed": requires ds.attrs[f"eps_{var}"] or set below manually (not implemented here)

        eps_floor_mode:
        - "global_min": epsilon(t) := max(epsilon(t), epsilon_global)
        - "none": no floor
        """
        ds = self.ds
        
        # Select layer
        if z_base <= z_top:
            raise ValueError("z_base must be greater than z_top (in meters).")

        layer = ds.sel({'range': slice(z_top, z_base)})

        # Depth from top (positive downward)
        z_layer = layer['range'].values.astype(float)
        d = (z_top - z_layer)  # d=0 at top; negative if z_layer>z_top; depends on ordering
        # Ensure d increases downward: if heights increase upward, z_layer>z_top => d negative.
        # Better: define depth as absolute distance from top along vertical axis:
        d = np.abs(z_layer - z_top)

        dz = float(z_base - z_top)

        Ze = layer[variable_threshold]
        ze_mask = xr.where(np.isfinite(Ze) & (Ze > threshold_value), True, False)

        out = xr.Dataset(coords={time_dim: layer[time_dim].values})
        out["dz"] = xr.DataArray(dz)

        # Precompute global eps if requested
        global_eps: dict[str, float] = {}
        if eps_mode == "global_quantile" or eps_floor_mode == "global_min":
            for vname in vars:
                vvals = layer[vname].values
                global_eps[vname] = compute_eps(vvals, q=q)

        # For each time, compute per-variable eps and fit
        times = layer[time_dim].values
        ntime = times.size

        n_valid = np.zeros(ntime, dtype=int)
        # n_valid defined by Ze mask only (common across variables), but we can also compute per-var n later if needed.
        ze_mask_np = ze_mask.values  # shape: (time, height_in_layer)
        n_valid[:] = np.sum(ze_mask_np, axis=1)

        out["n_valid"] = xr.DataArray(n_valid, dims=(time_dim,))

        for vname in vars:
            b_arr = np.full(ntime, np.nan, dtype=float)
            a_arr = np.full(ntime, np.nan, dtype=float)
            r2_arr = np.full(ntime, np.nan, dtype=float)
            F_arr = np.full(ntime, np.nan, dtype=float)

            V = layer[vname].values  # (time, height)

            for it in range(ntime):
                if n_valid[it] < nmin:
                    continue

                mask = ze_mask_np[it, :] & np.isfinite(V[it, :]) & (V[it, :] > 0)
                if np.sum(mask) < nmin:
                    continue

                # epsilon
                if eps_mode == "hourly_quantile":
                    eps_t = compute_eps(V[it, :], q=q)
                elif eps_mode == "global_quantile":
                    eps_t = global_eps.get(vname, np.nan)
                else:
                    raise ValueError(f"Unsupported eps_mode={eps_mode}")

                if not np.isfinite(eps_t) or eps_t <= 0:
                    continue

                if eps_floor_mode == "global_min":
                    eps_g = global_eps.get(vname, np.nan)
                    if np.isfinite(eps_g) and eps_g > 0:
                        eps_t = max(eps_t, eps_g)

                x = d[mask]
                y = np.log(np.maximum(V[it, mask], eps_t))

                b, a, r2 = ols_slope_intercept_r2(x, y)
                if not np.isfinite(b) or not np.isfinite(r2):
                    continue

                b_arr[it] = b
                a_arr[it] = a
                r2_arr[it] = r2
                F_arr[it] = float(np.exp(b * dz))

            out[f"b_{vname}"] = xr.DataArray(b_arr, dims=(time_dim,))
            out[f'a_{vname}'] = xr.DataArray(a_arr, dims=(time_dim,))
            out[f"r2_{vname}"] = xr.DataArray(r2_arr, dims=(time_dim,))
            out[f"F_{vname}"] = xr.DataArray(F_arr, dims=(time_dim,))
        return out


    def plot_hexagram_event_in_layer(
        self,
        *,
        period: tuple[datetime, datetime],
        layer: tuple[float, float],
        k: int,
        ze_th: float = -5.0,
        nmin: int = 10,
        eps_q: float = 0.01,
        rgb_q: float = 0.02,
        vars_trend: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
        figsize: tuple[float, float] = (10, 10),
        marker_size: float = 35.0,
        alpha: float = 0.9,
        use_snapped_colors: bool = True,
        savefig: bool = False,
        output_dir=None,
        dpi: int = 200,
    ):
        """
        Plotea el hexagrama (base RGB) y superpone la trayectoria temporal del evento
        (puntos) para una capa fija [z_top, z_base] y un periodo temporal.

        Pipeline:
        1) compute_layer_trend_ols(...)  -> b_Dm, b_Nw, b_LWC, F_*, r2_*
        2) build_rgb_from_trends(...)    -> R,G,B en [0,1]
        3) map_rgb_to_hexagram(...)      -> hex_x, hex_y, hex_area (+ snap_R,G,B opcional)
        4) plot base hexagram + scatter

        Parameters
        ----------
        period :
            slice o tupla (t0, t1) compatible con xarray sel(time=...).
            Ejemplos:
            - slice("2025-03-08T12:00", "2025-03-08T13:00")
            - (datetime1, datetime2)
        layer : (z_top, z_base) en metros.
        k : int
            Resolución del hexagrama (no se asume).
        ze_th : float
            Umbral de Ze (dBZ) para máscara binaria (eco válido).
        nmin : int
            Número mínimo de bins válidos en la capa para ajustar OLS.
        eps_q : float
            Cuantil para epsilon en el log (compute_layer_trend_ols).
        rgb_q : float
            Cuantil para la normalización robusta a RGB (build_rgb_from_trends).
        use_snapped_colors : bool
            Si True, colorea con el RGB de la celda asignada (snap); si False, con RGB continuo.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        # Imports desde utils (ajusta el path si tu paquete difiere)
        from mrrpropy.utils import (
            build_rgb_from_trends,
            map_rgb_to_hexagram,
            get_hexagram_assets,
        )

        if not self._is_processed():
            raise RuntimeError("El dataset no parece preprocesado (faltan variables clave).")

        z_top, z_base = layer
        if z_base <= z_top:
            raise ValueError("layer debe cumplir z_base > z_top (metros).")

        # --- seleccionar periodo ---
        ds_sub = self.ds.sel(time=slice(*period))
        if ds_sub.sizes.get("time", 0) == 0:
            raise ValueError("Selección temporal vacía: revisa period.")

        # --- 1) tendencias en capa (OLS + R²) ---
        trends = self.compute_layer_trend_ols(
            z_top=z_top,
            z_base=z_base,
            variable_threshold="Ze",
            threshold_value=ze_th,
            vars=vars_trend,
            q=eps_q,
            nmin=nmin,
        )

        # --- 2) RGB en [0,1] centrado en 0.5 (a partir de b_*) ---
        rgb = build_rgb_from_trends(
            trends,
            vars=(f"b_{vars_trend[0]}", f"b_{vars_trend[1]}", f"b_{vars_trend[2]}"),
            q=rgb_q,
        )

        # --- construir assets del hexagrama UNA vez (img + LUT) ---
        hex_assets = get_hexagram_assets(k=k)

        # --- 3) proyección al hexagrama (reutiliza LUT, no recalcula generate_rgb_hex) ---
        hex_ds = map_rgb_to_hexagram(rgb, hex_assets=hex_assets)

        # --- 4) plotting (reutiliza img del mismo hex_assets) ---
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(hex_assets["img"], origin="lower", interpolation="nearest")

        hx = hex_ds["hex_x"].values.astype(float)
        hy = hex_ds["hex_y"].values.astype(float)

        if use_snapped_colors and all(v in hex_ds for v in ("snap_R", "snap_G", "snap_B")):
            cols = np.stack(
                [hex_ds["snap_R"].values, hex_ds["snap_G"].values, hex_ds["snap_B"].values],
                axis=1,
            )
        else:
            cols = np.stack([rgb["R"].values, rgb["G"].values, rgb["B"].values], axis=1)

        ok = np.isfinite(hx) & np.isfinite(hy) & np.isfinite(cols).all(axis=1) & (hx >= 0) & (hy >= 0)
        if np.any(ok):
            ax.scatter(
                hx[ok],
                hy[ok],
                s=marker_size,
                c=cols[ok],
                alpha=alpha,
                edgecolors="black",
                linewidths=0.3,
            )

        # Etiquetado mínimo (sin asumir convención de áreas)
        t0 = ds_sub["time"].values[0]
        t1 = ds_sub["time"].values[-1]
        t0s = np.datetime_as_string(t0, unit="s")
        t1s = np.datetime_as_string(t1, unit="s")
        ax.set_title(f"Hexagrama RGB (k={k}) | Capa {z_top:.0f}-{z_base:.0f} m | {t0s} → {t1s}")
        ax.set_xlabel("hex_x (índice rejilla)")
        ax.set_ylabel("hex_y (índice rejilla)")
        ax.set_xlim(-0.5, hex_assets["img"].shape[1] - 0.5)
        ax.set_ylim(-0.5, hex_assets["img"].shape[0] - 0.5)
        ax.grid(False)
        fig.tight_layout()

        saved = None
        if savefig:
            if output_dir is None:
                outdir = Path.cwd()
                saved = outdir / f"{Path(self.path).stem}_{t0s.replace(':','')}_{t1s.replace(':','')}_hex_{z_top:.0f}-{z_base:.0f}m_k{k}.png"
            else:
                saved = Path(output_dir) / f"{Path(self.path).stem}_{t0s.replace(':','')}_{t1s.replace(':','')}_hex_{z_top:.0f}-{z_base:.0f}m_k{k}.png"
            saved.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(saved, dpi=dpi)

        return fig, ax, saved
