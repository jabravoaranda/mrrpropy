"""Shared rain-process metadata used across analysis, plotting, and hexagram code.

The canonical process vocabulary is defined by ``PROCESS_SIGNATURES``.  Legacy
labels are kept only as input aliases so old checkpoint CSV files can still be
read without creating duplicate process classes in new outputs.
"""

from __future__ import annotations

from typing import TypeAlias

ProcessSignature: TypeAlias = tuple[int, int, int]


PROCESS_SIGNATURES: dict[str, list[ProcessSignature]] = {
    # 0 -> 1 case -> Steady process
    "steady": [(0, 0, 0)],
    # 1 -> 6 cases -> Physically impossible
    # +1+0+0
    "+00": [(+1, 0, 0)],
    # +0+1+0
    "0+0": [(0, +1, 0)],
    # +0+0+1
    "00+": [(0, 0, +1)],
    # -1+0+0
    "-00": [(-1, 0, 0)],
    # +0-1+0
    "0-0": [(0, -1, 0)],
    # +0+0-1
    "00-": [(0, 0, -1)],
    # 2 -> 12 cases
    # +1+1+0
    "++0": [(+1, +1, 0)],
    # +1+0+1
    "growth": [(+1, 0, +1)],
    # +0+1+1 and +1+1+1
    "activation": [(0, +1, +1), (+1, +1, +1)],
    # -1+1+0
    "breakup": [(-1, +1, 0)],
    # -1+0+1
    "-0+": [(-1, 0, +1)],
    # +0-1+1
    "0-+": [(0, -1, +1)],
    # +1-1+0
    "coalescence": [(+1, -1, 0)],
    # +1+0-1
    "+0-": [(+1, 0, -1)],
    # +0+1-1
    "0+-": [(0, +1, -1)],
    # -1-1+0
    "--0": [(-1, -1, 0)],
    # -1+0-1
    "evaporation_weak": [(-1, 0, -1)],
    # +0-1-1
    "0--": [(0, -1, -1)],
    # 3 -> 8 cases
    # +1+1+1 is included in activation.
    # -1-1-1
    "evaporation_strong": [
        (-1, -1, -1),
    ],
    # +1-1-1
    "coalescence_loss": [(+1, -1, -1)],
    # -1+1-1
    "breakup_loss": [(-1, +1, -1)],
    # -1-1+1
    "--+": [(-1, -1, +1)],
    # +1-1+1
    "coalescence_gain": [(+1, -1, +1)],
    # -1+1+1
    "breakup_gain": [(-1, +1, +1)],
    # +1+1-1
    "++-": [(+1, +1, -1)],
}

PROCESS_CODES: dict[str, str] = {
    "unknown": "UNKNOWN",
    "steady_or_weak": "STEADY",
    "steady": "000",
    "+00": "+00",
    "0+0": "0+0",
    "00+": "00+",
    "-00": "-00",
    "0-0": "0-0",
    "00-": "00-",
    "++0": "++0",
    "+0-": "+0-",
    "0+-": "0+-",
    "--0": "--0",
    "-0+": "-0+",
    "0-+": "0-+",
    "--+": "--+",
    "++-": "++-",
    "0--": "0--",
    "breakup": "BREAKUP",
    "breakup_gain": "BU-GAIN",
    "breakup_loss": "BU-LOSS",
    "coalescence": "COAL.",
    "coalescence_gain": "COAL.-GAIN",
    "coalescence_loss": "COAL.-LOSS",
    "evaporation_weak": "EVAP.-WEAK",
    "evaporation_strong": "EVAP.-STRONG",
    "growth": "GROWTH",
    "activation": "ACTIV.",
    "no_data": "NO DATA",
}

PROCESS_ALIASES: dict[str, str] = {
    "growth_depletion": "coalescence",
    "growth_depletion_gain": "coalescence_gain",
    "growth_depletion_loss": "coalescence_loss",
    "condensation": "activation",
    "evaporation": "evaporation_strong",
}

PROCESS_ORDER: tuple[str, ...] = (
    "steady",
    "+00",
    "0+0",
    "00+",
    "-00",
    "0-0",
    "00-",
    "++0",
    "activation",
    "growth",
    "breakup",
    "-0+",
    "0-+",
    "coalescence",
    "+0-",
    "0+-",
    "--0",
    "evaporation_weak",
    "0--",
    "evaporation_strong",
    "coalescence_loss",
    "breakup_loss",
    "--+",
    "coalescence_gain",
    "breakup_gain",
    "++-",
    "steady_or_weak",
    "unknown",
    "no_data",
)

PROCESS_COLORS: dict[str, str] = {
    "steady": "#8f8f8f",
    "+00": "#c7a76c",
    "0+0": "#a6d854",
    "00+": "#fdb462",
    "-00": "#8dd3c7",
    "0-0": "#80b1d3",
    "00-": "#bebada",
    "++0": "#b3de69",
    "+0-": "#bc80bd",
    "0+-": "#fccde5",
    "--0": "#d9d9d9",
    "-0+": "#ccebc5",
    "0-+": "#ffed6f",
    "--+": "#1f78b4",
    "++-": "#b15928",
    "0--": "#6a3d9a",
    "breakup": "#12af54",
    "breakup_gain": "#13d7d7",
    "breakup_loss": "#24ca24",
    "coalescence": "#e31a1c",
    "coalescence_gain": "#fb9a99",
    "coalescence_loss": "#a50f15",
    "evaporation_weak": "#636363",
    "evaporation_strong": "#000000",
    "growth": "#91209b",
    "activation": "#66a61e",
    "steady_or_weak": "#8f8f8f",
    "unknown": "#666666",
    "no_data": "#bdbdbd",
}


def canonical_process_label(label: object) -> str:
    """Return the canonical rain-process label for a possibly legacy value."""
    text = str(label).strip()
    return PROCESS_ALIASES.get(text, text)


PROCESS_MARKERS: dict[str, str] = {
    "steady": ".",
    "+00": "x",
    "0+0": "x",
    "00+": "x",
    "-00": "x",
    "0-0": "x",
    "00-": "x",
    "++0": "x",
    # +1+0+1
    "growth": "v",
    # +0+1+1
    "activation": "^",
    # -1+1+0
    "breakup": "+",
    # -1+0+1
    "-0+": "x",
    # +0-1+1
    "0-+": "x",
    # +1-1+0
    "coalescence": "o",
    # +1+0-1
    "+0-": "x",
    # +0+1-1
    "0+-": "x",
    # -1-1+0
    "--0": "x",
    # -1+0-1
    "evaporation_weak": "d",
    # +0-1-1
    "0--": "8",
    # +1+1+1 is included in activation.
    # -1-1-1
    "evaporation_strong": "D",
    # +1-1-1
    "coalescence_loss": "p",
    # -1+1-1
    "breakup_loss": "h",
    # -1-1+1
    "--+": "x",
    # +1-1+1
    "coalescence_gain": "P",
    # -1+1+1
    "breakup_gain": "H",
    # +1+1-1
    "++-": "x",
    "no_data": ",",
}
