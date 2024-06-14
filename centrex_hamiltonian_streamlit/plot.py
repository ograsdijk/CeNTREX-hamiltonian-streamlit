import numpy as np
import numpy.typing as npt
import plotly
import plotly.graph_objects as go
from centrex_tlf import states
from centrex_tlf.states import ElectronicState


def generate_level_plot(
    states: list[states.CoupledState],
    reduced_hamiltonian: npt.NDArray[np.complex_],
    energy_scale: str = "kHz",
    states_highlight: list[states.CoupledState] = [],
) -> go.Figure:
    energy_scaling = {"kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    electronic = states[0].largest.electronic_state
    assert electronic is not None

    if len(states_highlight) == 0:
        states_highlight = states

    fig = go.Figure()

    x = np.array([-1 / 3, 1 / 3])

    energies = np.diag(reduced_hamiltonian.real) / (
        2 * np.pi * energy_scaling[energy_scale]
    )

    unique_F1_F = np.unique([(s.largest.F1, s.largest.F) for s in states], axis=0)

    colors = plotly.colors.qualitative.D3
    color_mapping = dict([(tuple(val), col) for val, col in zip(unique_F1_F, colors)])
    linestyle_mapping = {1: None, -1: "dash"}

    for state, energy in zip(states, energies):
        color = color_mapping[(state.largest.F1, state.largest.F)]
        name = (
            state.largest.state_string_custom(["F1", "F", "mF"])
            if state.largest.electronic_state == ElectronicState.X
            else state.largest.state_string_custom(["F1", "F", "mF", "P"])
        )
        parity = state.largest.P
        assert parity is not None
        if state in states_highlight:
            opacity = 1.0
        else:
            opacity = 0.3
        fig.add_trace(
            go.Scatter(
                x=state.largest.mF + x,
                y=[energy, energy],
                mode="lines",
                name=name,
                hoverinfo="y+name",
                marker=dict(color=color),
                line=dict(
                    width=5,
                    dash=linestyle_mapping[parity]
                    if state.largest.electronic_state == ElectronicState.B
                    else None,
                ),
                opacity=opacity,
            )
        )

    fig.update_layout(showlegend=False)
    fig.update_layout({"hoverlabel": {"namelength": -1}})
    fig.update_layout(xaxis=dict(tickfont=dict(size=20), titlefont=dict(size=20)))
    fig.update_layout(yaxis=dict(tickfont=dict(size=20), titlefont=dict(size=20)))
    fig.update_layout(
        xaxis_title="mF",
        yaxis_title=f"energy [{energy_scale}]",
        titlefont=dict(size=20),
    )
    fig.update_layout(hoverlabel=dict(font_size=16))
    fig.update_layout(
        title=dict(
            text=f"Levels for |{electronic.name}, J={states[0].largest.J}>",
        )
    )

    return fig


def generate_transition_level_plots(
    states_ground: list[states.CoupledState],
    reduced_hamiltonian_ground: npt.NDArray[np.complex_],
    states_excited: list[states.CoupledState],
    reduced_hamiltonian_excited: npt.NDArray[np.complex_],
    state_main_ground: states.CoupledState,
    state_main_excited: states.CoupledState,
    states_highlight: tuple[list[states.CoupledState], list[states.CoupledState]],
) -> tuple[go.Figure, go.Figure]:
    index_ground = states_ground.index(state_main_ground)
    index_excited = states_excited.index(state_main_excited)

    states_highlight_ground = states_highlight[0]
    states_highlight_excited = states_highlight[1]

    reduced_hamiltonian_ground -= (
        np.eye(reduced_hamiltonian_ground.shape[0])
        * reduced_hamiltonian_ground[index_ground, index_ground]
    )
    reduced_hamiltonian_excited -= (
        np.eye(reduced_hamiltonian_excited.shape[0])
        * reduced_hamiltonian_excited[index_excited, index_excited]
    )

    fig_ground = generate_level_plot(
        states_ground,
        reduced_hamiltonian_ground,
        states_highlight=states_highlight_ground,
    )
    fig_excited = generate_level_plot(
        states_excited,
        reduced_hamiltonian_excited,
        energy_scale="MHz"
        if state_main_excited.largest.electronic_state == ElectronicState.B
        else "kHz",
        states_highlight=states_highlight_excited,
    )

    return fig_ground, fig_excited
