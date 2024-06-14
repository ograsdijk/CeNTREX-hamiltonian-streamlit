from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from centrex_tlf import states


def generate_energy_dataframe(
    energies: npt.NDArray[np.floating],
    states: Sequence[states.CoupledState],
    energy_scale: str = "kHz",
) -> pd.DataFrame:
    energy_scaling = {"kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    states_column = [
        str(state.remove_small_components(5e-2))
        .replace(", I₁ = 1/2, I₂ = 1/2", "")
        .replace(", Ω = 1", "")
        .replace(", Ω = 0", "")
        .replace("+0.00j", "")
        .replace("-0.00j", "")
        for state in states
    ]

    df = pd.DataFrame(
        {
            "state": states_column,
            "energy": energies / (2 * np.pi * energy_scaling[energy_scale]),
        }
    )
    df = df.set_index("state")
    return df
