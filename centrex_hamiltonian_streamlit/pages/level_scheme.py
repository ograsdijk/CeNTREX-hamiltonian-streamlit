import numpy as np
import streamlit as st
from centrex_tlf import states

from ..dataframe import generate_energy_dataframe
from ..hamiltonian import generate_hamiltonian
from ..plot import generate_level_plot


def show_level_scheme():
    col_select, col_fig = st.columns([1, 3])
    with col_select:
        # with st.expander("Manifold selection"):
        electronic = st.selectbox(
            label="Electronic", options=[s for s in states.ElectronicState], index=0
        )
        assert electronic is not None
        J_options = (
            [0, 1, 2, 3, 4, 5, 6]
            if electronic == states.ElectronicState.X
            else [1, 2, 3, 4]
        )
        J = st.selectbox(label="J", options=J_options, index=0)
        assert J is not None

        F1_options = np.arange(
            J - 1 / 2 if J > 0 else 1 / 2, J + 3 / 2, dtype=np.float64
        )
        F1 = st.multiselect(label="F1", options=F1_options, default=F1_options)
        F_options = np.arange(J - 1 if J > 0 else 0, J + 2, 1, dtype=np.int64)
        F = st.multiselect(label="F", options=F_options, default=F_options)

        E = st.number_input(
            label="E [V/cm]",
            min_value=0,
            step=1,
            placeholder="electric field in V/cm",
        )
        B = st.number_input(
            label="B [G]",
            min_value=0.0,
            step=1e-2,
            placeholder="magnetic field in G",
            value=1e-2,
        )

    coupled_states, ham = generate_hamiltonian(electronic, J, F1, F, E, B)

    ham -= np.eye(ham.shape[0]) * ham[0, 0]

    fig = generate_level_plot(
        coupled_states,
        ham,
        energy_scale="kHz" if electronic == states.ElectronicState.X else "MHz",
    )
    energy_labeling = {states.ElectronicState.X: "kHz", states.ElectronicState.B: "MHz"}

    df = generate_energy_dataframe(
        np.diag(ham).real, coupled_states, energy_labeling[electronic]
    )
    format = "%.3f" if electronic == states.ElectronicState.B else "%.0f"

    with col_fig:
        st.plotly_chart(fig)

        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "energy": st.column_config.NumberColumn(
                    f"energy [{energy_labeling[electronic]}]", format=format
                )
            },
        )
