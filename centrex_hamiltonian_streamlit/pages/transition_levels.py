import numpy as np
import streamlit as st
from centrex_tlf import couplings, states

from ..hamiltonian import generate_hamiltonian
from ..plot import generate_transition_level_plots


def show_transition_levels():
    with st.expander("Transition selection"):
        st.header("Fields")
        colf, colc = st.columns(2)
        with colf:
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
        with colc:
            polarization_str = st.selectbox(
                label="Polarization", options="Z,X,Y,σm,σp".split(",")
            )
            assert polarization_str is not None
            polarization_mapping = {
                "Z": couplings.polarization_Z,
                "X": couplings.polarization_X,
                "Y": couplings.polarization_Y,
                "σm": couplings.polarization_σm,
                "σp": couplings.polarization_σp,
            }
            polarization = polarization_mapping[polarization_str]

        colg, cole = st.columns(2)
        with colg:
            st.header("Ground")
            Jg_options = [0, 1, 2, 3, 4, 5, 6]
            Jg = st.selectbox(label="J", options=Jg_options, index=0)
            assert Jg is not None

            F1g_options = np.arange(Jg - 1 / 2 if Jg > 0 else 1 / 2, Jg + 3 / 2)
            F1g = st.selectbox(label="F1", options=F1g_options, index=0, key="F1g")
            assert F1g is not None

            Fg_options = np.arange(
                F1g - 1 / 2 if F1g - 1 / 2 > 0 else 0, F1g + 1, 1, dtype=int
            )
            Fg = st.selectbox(label="F", options=Fg_options, index=0, key="Fg")
            assert Fg is not None

            mFg_options = np.arange(-Fg, Fg + 1, dtype=int)
            mFg = st.selectbox(
                label="mF", options=mFg_options, index=len(mFg_options) // 2
            )

        with cole:
            st.header("Excited")
            electronic_excited = st.selectbox(
                label="Electronic", options=[s for s in states.ElectronicState], index=1
            )
            assert electronic_excited is not None
            if electronic_excited == states.ElectronicState.B:
                Je_options = [1, 2, 3, 4]
            else:
                Je_options = [0, 1, 2, 3, 4, 5, 6]

            Je = st.selectbox(label="J", options=Je_options, index=0, key="Je")
            assert Je is not None

            F1e_options = np.arange(Je - 1 / 2 if Je > 0 else 1 / 2, Je + 3 / 2)
            F1e = st.selectbox(label="F1", options=F1e_options, index=0, key="F1e")
            assert F1e is not None

            Fe_options = np.arange(
                F1e - 1 / 2 if F1e - 1 / 2 > 0 else 0, F1e + 1, 1, dtype=int
            )
            Fe = st.selectbox(label="F", options=Fe_options, index=0, key="Fe")
            assert Fe is not None
            mFe_options = np.arange(-Fe, Fe + 1, dtype=int)
            mFe = st.selectbox(
                label="mF", options=mFe_options, index=len(mFe_options) // 2, key="mFe"
            )

    coupled_states_ground, ham_ground = generate_hamiltonian(
        states.ElectronicState.X,
        Jg,
        np.arange(Jg - 1 / 2 if Jg > 0 else 1 / 2, Jg + 3 / 2),
        np.arange(Jg - 1 if Jg > 0 else 0, Jg + 2).astype(int),
        E,
        B,
    )

    if electronic_excited == states.ElectronicState.B:
        coupled_states_excited, ham_excited = generate_hamiltonian(
            states.ElectronicState.B, Je, F1e, Fe, E, B, P=(-1) ** (Jg + 1)
        )
    else:
        coupled_states_excited, ham_excited = generate_hamiltonian(
            states.ElectronicState.X,
            Je,
            np.arange(Je - 1 / 2 if Je > 0 else 1 / 2, Je + 3 / 2),
            np.arange(Je - 1 if Je > 0 else 0, Je + 2, dtype=int),
            E,
            B,
            P=(-1) ** (Jg + 1),
        )

    smg = states.CoupledBasisState(
        J=Jg,
        F1=F1g,
        F=Fg,
        mF=mFg,
        I1=1 / 2,
        I2=1 / 2,
        P=(-1) ** Jg,
        electronic_state=states.ElectronicState.X,
    )
    sme = states.CoupledBasisState(
        J=Je,
        F1=F1e,
        F=Fe,
        I1=1 / 2,
        I2=1 / 2,
        P=(-1) ** (Jg + 1),
        mF=mFe,
        Ω=1 if electronic_excited == states.ElectronicState.B else 0,
        electronic_state=electronic_excited,
    )

    # generate the coupling matrix between the two manifolds
    amps = [[a for a, si in state if smg @ si != 0] for state in coupled_states_ground]
    idx = np.argmax([a[0] if len(a) > 0 else 0 for a in amps])
    state_main_ground = coupled_states_ground[idx]
    amps = [[a for a, si in state if sme @ si != 0] for state in coupled_states_excited]
    idx = np.argmax([a[0] if len(a) > 0 else 0 for a in amps])
    state_main_excited = coupled_states_excited[idx]

    # ME_main = couplings.calculate_ED_ME_mixed_state(
    #     state_main_excited,
    #     state_main_ground,
    #     pol_vec=polarization.vector,
    #     normalize_pol=True,
    # )

    matrix = np.abs(
        couplings.generate_coupling_matrix(
            coupled_states_ground + coupled_states_excited,
            coupled_states_ground,
            coupled_states_excited,
            polarization.vector,
            reduced=False,
        ).real
    )
    matrix[matrix < 1e-2] = 0

    indices_nonzero = np.nonzero(np.triu(matrix))
    coupled_ground = [
        coupled_states_ground[idx] for idx in np.unique(indices_nonzero[0])
    ]
    coupled_excited = [
        coupled_states_excited[idx]
        for idx in np.unique(indices_nonzero[1] - len(coupled_states_ground))
    ]

    fig_ground, fig_excited = generate_transition_level_plots(
        coupled_states_ground,
        ham_ground,
        coupled_states_excited,
        ham_excited,
        state_main_ground,
        state_main_excited,
        states_highlight=(coupled_ground, coupled_excited),
    )

    st.plotly_chart(fig_excited)
    st.plotly_chart(fig_ground)
