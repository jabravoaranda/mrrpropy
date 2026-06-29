"""Shared rain-process metadata used across analysis, plotting, and hexagram code."""

from __future__ import annotations

from typing import TypeAlias

ProcessSignature: TypeAlias = tuple[int, int, int]


PROCESS_SIGNATURES: dict[str, list[ProcessSignature]] = {
    "breakup": [(-1, +1, 0)],
    "growth_depletion": [(+1, -1, 0)],
    "growth_depletion_gain": [(+1, -1, +1)],
    "growth_depletion_loss": [(+1, -1, -1)],
    "evaporation": [(-1, -1, -1), (-1, -1, 0), (-1, 0, -1)],
    "growth": [(+1, 0, +1)],
    "activation": [(+1, +1, +1), (0, +1, +1)],
}


PROCESS_CODES: dict[str, str] = {
    "unknown": "UNKNOWN",
    "steady_or_weak": "STEADY",
    "breakup": "BREAKUP",
    "growth_depletion": "GROWTH-DEPLETION",
    "growth_depletion_gain": "GD-GAIN",
    "growth_depletion_loss": "GD-LOSS",
    "evaporation": "EVAP.",
    "growth": "GROWTH",
    "activation": "ACTIV.",
}


PROCESS_MARKERS: dict[str, str] = {
    "breakup": "s",
    "growth_depletion": "o",
    "growth_depletion_gain": "D",
    "growth_depletion_loss": "d",
    "evaporation": "v",
    "growth": "^",
    "activation": "+",
    "steady_or_weak": ".",
    "unknown": ".",
    "no_data": ".",
}
