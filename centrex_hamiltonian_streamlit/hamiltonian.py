import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
from centrex_tlf import hamiltonian, states

data_path = Path(__file__).parent


@dataclass
class HamiltonianX:
    uncoupled_states: list[states.UncoupledBasisState]
    coupled_states: list[states.CoupledBasisState]
    hamiltonian: hamiltonian.HamiltonianUncoupledX
    transform: npt.NDArray[np.complex_]


def generate_hamiltonian(
    electronic: states.ElectronicState, J: int, E: float, B: float = 1e-3
) -> tuple[List[states.CoupledState], npt.NDArray[np.complex_]]:
    Evec = np.array([0, 0, E])
    Bvec = np.array([0, 0, B])

    selector = states.QuantumSelector(J=J, electronic=electronic)

    if electronic == states.ElectronicState.X:
        with open(data_path / "X_state_ham.pkl", "rb") as f:
            hamiltonian_x = HamiltonianX(**pickle.load(f))
            hamiltonian_func = hamiltonian.generate_uncoupled_hamiltonian_X_function(
                hamiltonian_x.hamiltonian
            )
            ham = (
                hamiltonian_x.transform.conj().T
                @ hamiltonian_func(Evec, Bvec)
                @ hamiltonian_x.transform
            )
            ham_diag = hamiltonian.generate_diagonalized_hamiltonian(
                ham, keep_order=True
            )
            states_diag = hamiltonian.matrix_to_states(
                ham_diag.V, list(hamiltonian_x.coupled_states)
            )
            selected_states = states.find_exact_states(
                [1 * state for state in states.generate_coupled_states_X(selector)],
                list(hamiltonian_x.coupled_states),
                states_diag,
                ham_diag.H,
                ham_diag.V,
            )
            selected_states = [
                state.remove_small_components(1e-3) for state in selected_states
            ]

            ham_reduced = hamiltonian.reduced_basis_hamiltonian(
                states_diag, ham_diag.H, selected_states
            )

    # elif electronic == states.ElectronicState.B:
    #     coupled_states = states.generate_coupled_states_B(selector)
    #     ham = hamiltonian.generate_reduced_B_hamiltonian(coupled_states, E=Evec, B=Bvec)

    return selected_states, ham_reduced
