import streamlit as st
from centrex_tlf import states
from hamiltonian import generate_hamiltonian
from plot import generate_level_plot

with st.sidebar:
    st.title("Manifold selection")
    electronic = st.selectbox(
        label="Electronic", options=[s for s in states.ElectronicState], index=0
    )
    J = st.selectbox(label="J", options=[0, 1, 2, 3, 4, 5, 6], index=0)

    E = st.number_input(
        label="E [V/cm]", min_value=0, step=1, placeholder="electric field in V/cm"
    )
    B = st.number_input(
        label="B [G]",
        min_value=0.0,
        step=1e-2,
        placeholder="magnetic field in G",
        value=1e-2,
    )

coupled_states, ham = generate_hamiltonian(electronic, J, E, B)

fig = generate_level_plot(coupled_states, ham)

st.plotly_chart(fig)
