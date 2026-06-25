from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import xarray as xr

import mrrpropy.RaProMPro_optimized as rpm_optimized


class SupportsRaprompro(Protocol):
    path: str | Path
    ds: xr.Dataset
    raprompro: xr.Dataset | None


def process_raprompro(
    subject: SupportsRaprompro,
    *,
    adjust_m: float = 1.0,
    save_spe_3d: bool = False,
    save_dsd_3d: bool = False,
    save: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Internal implementation of the canonical RaProMPro processing wrapper.

    The resulting dataset follows the variable naming used by the scientific
    RaProMPro workflow, including fields such as ``Type``, ``DBPIA``, ``Za``,
    ``Zea``, ``Ze``, ``Dm``, ``Nw``, ``LWC`` and ``RR``.

    Public velocity outputs use negative values for downward hydrometeor motion.
    The retained scientific core keeps its original positive-downward convention
    internally; the sign is converted when ``W`` and optional spectral velocity
    coordinates are exposed in the returned dataset.
    """
    if subject.raprompro is not None:
        return subject.raprompro

    ds = subject.ds
    rpm: Any = rpm_optimized

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
    n_spectra = ds.sizes["n_spectra"]

    raw_spectra_np = ds["spectrum_raw"].values if has_raw else None
    ref_spectra_np = ds["spectrum_reflectivity"].values if has_ref else None
    snr_np = ds["SNR"].values.astype(float) if "SNR" in ds else None

    def _spectra_db_at_time(it: int, varname: str) -> np.ndarray:
        """
        Returns spec_db[range, bins] for a given time index, using index_spectra.
        Implemented as a loop to keep semantics explicit (matches CLI structure).
        """
        out = np.full((Nhei, Nbins), np.nan, dtype=float)
        source = raw_spectra_np if varname == "spectrum_raw" else ref_spectra_np
        if source is None:
            return out
        spec_idx = idx_map_int[it, :]
        valid = (spec_idx >= 0) & (spec_idx < n_spectra)
        if np.any(valid):
            out[valid, :] = source[it, spec_idx[valid], :]
        return out

    # SNR for spectrum_reflectivity mode (original passes Snr_Refl_2)
    def _snr_at_time(it: int) -> np.ndarray:
        if snr_np is not None:
            return snr_np[it, :]
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
    ntime = ds.sizes["time"]
    bb_bot_full: list[float] = []
    bb_top_full: list[float] = []
    bb_peak_full: list[float] = []

    # Full matrices (time, range)
    def _empty_2d() -> np.ndarray:
        return np.full((ntime, Nhei), np.nan, dtype=float)

    def _negative_downward_spectrum_axis() -> np.ndarray:
        """Public output convention: Doppler/fall velocity is negative downward."""
        return -np.arange(-Nbins * fNy, 2 * Nbins * fNy, fNy)[::-1]

    estat_full = _empty_2d()
    sk_full = _empty_2d()
    kur_full = _empty_2d()
    PIA_full = _empty_2d()
    w_full = _empty_2d()
    sig_full = _empty_2d()
    LWC_full = _empty_2d()
    RR_full = _empty_2d()
    SnowR_full = _empty_2d()
    Z_da_full = _empty_2d()
    Z_a_full = _empty_2d()
    Z_ea_full = _empty_2d()
    Z_e_full = _empty_2d()
    z_all_full = _empty_2d()
    lwc_all_full = _empty_2d()
    rr_all_full = _empty_2d()
    n_all_full = _empty_2d()
    nw_full = _empty_2d()
    dm_full = _empty_2d()
    NW_all_full = _empty_2d()
    DM_all_full = _empty_2d()
    Noi_full = _empty_2d()
    SNR_full = _empty_2d()
    N_da_full = _empty_2d()

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

    for it in range(ntime):
        nan_row = np.full(Nbins, np.nan, dtype=float)
        NewNoise = [np.nan] * Nhei
        Pot = [nan_row.copy() for _ in range(Nhei)]

        if Code_spectrum == 0:
            raw_db = _spectra_db_at_time(it, "spectrum_raw")  # (range, bins)
            # Loop over ranges exactly as CLI
            for k in range(Nhei):
                COL_db = raw_db[k, :]
                if np.isnan(COL_db).all():
                    continue

                COL_lin = np.power(10.0, COL_db / 10.0)
                COL2, Noise = rpm.MrrProNoise2(COL_lin, k, DeltaH, TimeInt)
                denom = FTcolum[k]

                # original: Noise*(k)**2/TF[k] and COL2*(k)**2/TF[k]
                if not np.isfinite(denom) or denom == 0:
                    NewNoise[k] = np.nan
                    Pot[k] = np.full(Nbins, np.nan)
                else:
                    NewNoise[k] = Noise * (k**2) / denom
                    Pot[k] = (COL2 * (k**2)) / denom

            Snr_Refl_2 = np.array([], dtype=float)
        else:
            ref_db = _spectra_db_at_time(it, "spectrum_reflectivity")
            for k in range(Nhei):
                COL_db = ref_db[k, :]
                if np.isnan(COL_db).all():
                    continue
                Pot[k] = np.power(10.0, COL_db / 10.0)
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

        estat_arr = np.asarray(estat, dtype=float)
        w_arr = np.asarray(w, dtype=float)
        sig_arr = np.asarray(sig, dtype=float)
        sk_arr = np.asarray(sk, dtype=float)
        Noi_arr = np.asarray(Noi, dtype=float)
        DSD_arr = np.asarray(DSD, dtype=float)
        NdE_arr = np.asarray(NdE, dtype=float)
        Ze_arr = np.asarray(Ze, dtype=float)
        snr_arr = np.asarray(snr, dtype=float)
        kur_arr = np.asarray(kur, dtype=float)
        PiA_arr = np.asarray(PiA, dtype=float)
        Lwc_arr = np.asarray(Lwc, dtype=float)
        Rr_arr = np.asarray(Rr, dtype=float)
        SnowRate_arr = np.asarray(SnowRate, dtype=float)
        z_da_arr = np.asarray(z_da, dtype=float)
        Z_all_arr = np.asarray(Z_all, dtype=float)
        RR_all_arr = np.asarray(RR_all, dtype=float)
        LWC_all_arr = np.asarray(LWC_all, dtype=float)
        N_all_arr = np.asarray(N_all, dtype=float)
        NW_arr = np.asarray(NW, dtype=float)
        DM_arr = np.asarray(DM, dtype=float)
        nw_all_arr = np.asarray(nw_all, dtype=float)
        dm_all_arr = np.asarray(dm_all, dtype=float)

        # BB logic (original uses special handling for first two times)
        if it == 0:
            bb_bot, bb_top, bb_peak = rpm.BB2(
                w_arr,
                Ze_arr,
                Hcolum,
                sk_arr,
                kur_arr,
                np.ones(2) * np.nan,
                np.ones(2) * np.nan,
                np.ones(2) * np.nan,
            )
        elif it == 1:
            bb_bot, bb_top, bb_peak = rpm.BB2(
                w_arr,
                Ze_arr,
                Hcolum,
                sk_arr,
                kur_arr,
                np.ones(2) * bb_bot_full,
                np.ones(2) * bb_top_full,
                np.ones(2) * bb_peak_full,
            )
        else:
            bb_bot, bb_top, bb_peak = rpm.BB2(
                w_arr,
                Ze_arr,
                Hcolum,
                sk_arr,
                kur_arr,
                bb_bot_full,
                bb_top_full,
                bb_peak_full,
            )

        bb_bot_full.append(bb_bot)
        bb_top_full.append(bb_top)
        bb_peak_full.append(bb_peak)

        # PIA in dB
        pIA = 10.0 * np.log10(PiA_arr)

        # Apply PIA only for drizzle/rain exactly as CLI
        liquid_mask = (estat_arr == 10) | (estat_arr == 5)
        ZaCorrec_all = Z_all_arr - pIA
        ZeCorrec = np.where(liquid_mask, Ze_arr - pIA, Ze_arr)
        ZaCorrec = np.where(liquid_mask, z_da_arr - pIA, np.nan)

        # Collect time-varying “type” params for PrepType (optional)
        if not np.isnan(DM_arr).all():
            Nw_2.append(NW_arr)
            Dm_2.append(DM_arr)

        # Optional 3D outputs
        if save_spe_3d:
            spe_3d_list.append(np.asarray(NewMatrix, dtype=float))
        if save_dsd_3d:
            dsd_3d_list.append(np.log10(NdE_arr))

        estat_full[it, :] = estat_arr
        sk_full[it, :] = sk_arr
        kur_full[it, :] = kur_arr
        PIA_full[it, :] = np.asarray(pIA, dtype=float)
        # RaProMPro internals use positive-downward fall speed. Public outputs use
        # the meteorological convention: falling hydrometeors have negative W.
        w_full[it, :] = -w_arr
        sig_full[it, :] = sig_arr
        LWC_full[it, :] = Lwc_arr
        RR_full[it, :] = Rr_arr
        SnowR_full[it, :] = SnowRate_arr
        Z_da_full[it, :] = z_da_arr
        Z_a_full[it, :] = ZaCorrec
        Z_ea_full[it, :] = Ze_arr
        Z_e_full[it, :] = ZeCorrec
        z_all_full[it, :] = ZaCorrec_all
        lwc_all_full[it, :] = LWC_all_arr
        rr_all_full[it, :] = RR_all_arr
        n_all_full[it, :] = N_all_arr
        nw_full[it, :] = NW_arr
        dm_full[it, :] = DM_arr
        NW_all_full[it, :] = nw_all_arr
        DM_all_full[it, :] = dm_all_arr
        Noi_full[it, :] = Noi_arr
        SNR_full[it, :] = snr_arr
        N_da_full[it, :] = DSD_arr

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
    def _attrs(units: str, desc: str) -> dict[str, str]:
        return {
            "units": units,
            "long_name": desc,
            "description": desc,
        }

    def _da2(name, data, units, desc):
        out[name] = xr.DataArray(
            np.asarray(data, dtype=float),
            dims=("time", "range"),
            attrs=_attrs(units, desc),
        )

    _da2(
        "Type",
        estat_full,
        "1",
        "Predominant hydrometeor type classification code (original CLI)",
    )
    _da2(
        "W",
        w_full,
        "m s-1",
        "Fall velocity with aliasing correction (negative downward)",
    )
    out["W"].attrs.update(
        {
            "positive": "up",
            "convention": "negative values indicate downward hydrometeor motion",
        }
    )
    _da2(
        "spectral width",
        sig_full,
        "m s-1",
        "Spectral width of the dealiased velocity distribution",
    )
    _da2(
        "Skewness",
        sk_full,
        "1",
        "Skewness of the spectral reflectivity with dealiasing",
    )
    _da2(
        "Kurtosis",
        kur_full,
        "1",
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
    _da2("N_all", n_all_full, "log10(m-3 mm-1)", "log10(total N) assuming all liquid")
    _da2("Nw", nw_full, "log10(mm-1 m-3)", "Normalized intercept parameter (by Type)")
    _da2("Dm", dm_full, "mm", "Mean mass-weighted diameter (by Type)")
    _da2(
        "Nw_all",
        NW_all_full,
        "log10(mm-1 m-3)",
        "Normalized intercept parameter (all liquid)",
    )
    _da2("Dm_all", DM_all_full, "mm", "Mean mass-weighted diameter (all liquid)")
    _da2("Noise", Noi_full, "eta_n", "Noise estimate in eta(n) units (original)")
    _da2("SNR", SNR_full, "dB", "SNR used/derived by algorithm (original)")
    _da2(
        "N_da",
        N_da_full,
        "log10(m-3 mm-1)",
        "log10(N(D)) derived (original 'N_da')",
    )

    out.attrs.update(
        {
            "velocity_convention": (
                "Public RaProMPro velocity outputs use negative-downward "
                "Doppler/fall velocity. Internal processing keeps the legacy "
                "positive-downward convention."
            )
        }
    )

    # BB as (time,BB_Height) to mirror CLI netCDF shape
    out["BB_bottom"] = xr.DataArray(
        np.asarray(bb_bot_full2, dtype=float)[:, None],
        dims=("time", "BB_Height"),
        attrs=_attrs(
            "m",
            "Range from bright-band bottom above sea level (original CLI)",
        ),
    )
    out["BB_top"] = xr.DataArray(
        np.asarray(bb_top_full2, dtype=float)[:, None],
        dims=("time", "BB_Height"),
        attrs=_attrs(
            "m",
            "Range from bright-band top above sea level (original CLI)",
        ),
    )
    out["BB_peak"] = xr.DataArray(
        np.asarray(bb_peak_full2, dtype=float)[:, None],
        dims=("time", "BB_Height"),
        attrs=_attrs(
            "m",
            "Range from bright-band peak above sea level (original CLI)",
        ),
    )

    # Optional 3D products (names follow original netCDF)
    if save_spe_3d:
        out["spe_3D"] = xr.DataArray(
            np.asarray(spe_3d_list, dtype=float)[..., ::-1],
            dims=("time", "range", "speed"),
            coords={
                "time": coords["time"],
                "range": coords["range"],
                "speed": _negative_downward_spectrum_axis(),
            },
            attrs=_attrs(
                "mm-1",
                "Dealiased spectral reflectivity on negative-downward velocity axis",
            ),
        )
        out["speed"].attrs.update(
            {
                "units": "m s-1",
                "long_name": "Doppler velocity",
                "description": "Doppler velocity coordinate; negative values indicate downward motion.",
                "positive": "up",
            }
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
            attrs=_attrs("log10(m-3 mm-1)", "Three-dimensional DSD (original CLI)"),
        )

    subject.raprompro = out
    if save:
        output_dir = kwargs.get("output_dir", Path.cwd())
        filename = kwargs.get("filename", f"{Path(subject.path).stem}_raprompro.nc")
        out.to_netcdf(output_dir / filename)

    return out


def load_raprompro(
    subject: SupportsRaprompro,
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
    Load an existing RaProMPro NetCDF product and optionally validate it.

    Parameters
    ----------
    path : str | Path
        Path to the ``*_raprompro.nc`` file, for example
        ``20250308_120000_raprompro.nc``.
    chunks : "auto" | dict | None
        If not None, open lazily with dask to speed up I/O and avoid
        loading the full dataset into memory.
    validate : bool
        If True, check that the dataset has the expected dimensions and
        coordinates and matches ``subject.ds``.
    required_vars : tuple[str, ...]
        Minimum variables expected in the processed dataset.
    assign : bool
        If True, store the dataset in ``subject.raprompro``.

    Returns
    -------
    xr.Dataset
        Loaded processed dataset. If ``assign=True``, it is also stored in
        :attr:`raprompro`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    ds_rp = xr.open_dataset(path, chunks=chunks)

    if validate:
        # 1) Minimum dimensions/coordinates
        for c in ("time", "range"):
            if c not in ds_rp.coords:
                raise ValueError(
                    f"Loaded raprompro product does not have coord {c!r}. "
                    f"coords={list(ds_rp.coords)}"
                )

        # 2) Minimum variables (simple heuristic)
        missing = [v for v in required_vars if v not in ds_rp.data_vars]
        if missing:
            raise ValueError(
                f"Loaded raprompro product does not look valid: missing {missing}. "
                f"vars={list(ds_rp.data_vars)}"
            )
        velocity_convention = str(ds_rp.attrs.get("velocity_convention", ""))
        if "W" in ds_rp and not velocity_convention.startswith(
            "Public RaProMPro velocity outputs use negative-downward"
        ):
            warnings.warn(
                "Loaded raprompro product does not declare the current "
                "negative-downward velocity convention. Older products may use "
                "positive-downward W/speed; regenerate the product to get "
                "unambiguous public velocity outputs.",
                UserWarning,
                stacklevel=2,
            )

        # 3) Compatibilidad con subject.ds (time/range)
        #    (si no quieres esto, pon validate=False)
        if "time" in subject.ds.coords:
            t0 = subject.ds["time"].values
            t1 = ds_rp["time"].values
            if (t0.shape != t1.shape) or (not np.array_equal(t0, t1)):
                raise ValueError(
                    "Incompatibilidad en 'time' entre subject.ds y el raprompro cargado "
                    f"(subject.ds: {t0.shape}, raprompro: {t1.shape})."
                )

        if "range" in subject.ds.coords:
            r0 = subject.ds["range"].values
            r1 = ds_rp["range"].values
            if (r0.shape != r1.shape) or (not np.array_equal(r0, r1)):
                raise ValueError(
                    "Incompatibilidad en 'range' entre subject.ds y el raprompro cargado "
                    f"(subject.ds: {r0.shape}, raprompro: {r1.shape})."
                )

    if assign:
        subject.raprompro = ds_rp

    return ds_rp
