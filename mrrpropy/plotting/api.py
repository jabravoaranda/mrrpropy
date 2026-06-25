"""Public plotting interface exposed as ``MRRProData.plot``."""

from __future__ import annotations

from typing import Any

from mrrpropy.hexagram import plot_process_to_hexagram
from mrrpropy.plotting import dsd, profiles, quicklook, rain_processes, spectra


class RainPlotAPI:
    """Rain-process plotting namespace available as ``mrr.plot.rain``."""

    def __init__(self, subject: Any) -> None:
        self._subject = subject

    def layer_2d(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_rain_process_in_layer_2d(
            self._subject, *args, **kwargs
        )

    def layer_hexagram(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_rain_process_in_layer_hexagram(
            self._subject, *args, **kwargs
        )

    def event_scatter(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_event_scatter(self._subject, *args, **kwargs)

    def region_scatter(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_region_scatter(self._subject, *args, **kwargs)

    def process_scatter(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_process_scatter(self._subject, *args, **kwargs)

    def scan_scatter_compare(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_scan_process_scatter_compare(
            self._subject, *args, **kwargs
        )

    def event_vertical_profiles(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_event_vertical_percent_profiles(
            self._subject, *args, **kwargs
        )

    def process_vertical_profiles(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_process_vertical_percent_profiles(
            self._subject, *args, **kwargs
        )

    def evolution(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_processes_evolution(self._subject, *args, **kwargs)

    def classified_hexagram(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_classified_processes_on_hexagram(
            self._subject, *args, **kwargs
        )

    def column_scan(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_column_process_scan(self._subject, *args, **kwargs)

    def fused_quicklook(self, *args: Any, **kwargs: Any) -> Any:
        return rain_processes.plot_fused_process_quicklook(*args, **kwargs)

    def process_mask_hexagram(self, *args: Any, **kwargs: Any) -> Any:
        return plot_process_to_hexagram(*args, **kwargs)


class PlotAPI:
    """Plotting namespace available as ``mrr.plot``."""

    def __init__(self, subject: Any) -> None:
        self._subject = subject
        self.rain = RainPlotAPI(subject)

    def quicklook(self, *args: Any, **kwargs: Any) -> Any:
        return quicklook.quicklook(self._subject, *args, **kwargs)

    def spectrum(self, *args: Any, **kwargs: Any) -> Any:
        return spectra.plot_spectrum(self._subject, *args, **kwargs)

    def spectra_by_range(self, *args: Any, **kwargs: Any) -> Any:
        return spectra.plot_spectra_by_range(self._subject, *args, **kwargs)

    def spectrogram(self, *args: Any, **kwargs: Any) -> Any:
        return spectra.plot_spectrogram(self._subject, *args, **kwargs)

    def dsdgram(self, *args: Any, **kwargs: Any) -> Any:
        return dsd.plot_dsdgram(self._subject, *args, **kwargs)

    def dsd_by_range(self, *args: Any, **kwargs: Any) -> Any:
        return dsd.plot_dsd_by_range(self._subject, *args, **kwargs)

    def microphysical_profiles(self, *args: Any, **kwargs: Any) -> Any:
        return profiles.plot_microphysical_properties_profiles(
            self._subject, *args, **kwargs
        )
