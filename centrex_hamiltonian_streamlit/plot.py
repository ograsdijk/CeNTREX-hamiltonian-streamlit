import numpy as np
import numpy.typing as npt
import plotly
import plotly.graph_objects as go
from centrex_tlf import states


def generate_level_plot(
    states: list[states.CoupledState],
    reduced_hamiltonian: npt.NDArray[np.complex_],
) -> go.Figure:
    fig = go.Figure()

    x = np.array([-1 / 3, 1 / 3])

    energies = np.diag(reduced_hamiltonian.real) / (2 * np.pi * 1e3)
    energies -= energies[0]

    unique_F1_F = np.unique([(s.largest.F1, s.largest.F) for s in states], axis=0)

    colors = plotly.colors.qualitative.D3
    color_mapping = dict([(tuple(val), col) for val, col in zip(unique_F1_F, colors)])

    for state, energy in zip(states, energies):
        color = color_mapping[(state.largest.F1, state.largest.F)]
        fig.add_trace(
            go.Scatter(
                x=state.largest.mF + x,
                y=[energy, energy],
                mode="lines",
                name=state.largest.state_string_custom(["F1", "F", "mF"]),
                hoverinfo="y+name",
                marker=dict(color=color),
                line=dict(width=5),
            )
        )

    fig.update_layout(showlegend=False)
    fig.update_layout({"hoverlabel": {"namelength": -1}})
    fig.update_layout(xaxis=dict(tickfont=dict(size=20), titlefont=dict(size=20)))
    fig.update_layout(yaxis=dict(tickfont=dict(size=20), titlefont=dict(size=20)))
    fig.update_layout(
        xaxis_title="mF", yaxis_title="energy [kHz]", titlefont=dict(size=20)
    )
    fig.update_layout(hoverlabel=dict(font_size=16))
    fig.update_layout(
        title=dict(
            text=f"Levels for |{states[0].largest.electronic_state.name}, J={states[0].largest.J}>",
        )
    )

    return fig
