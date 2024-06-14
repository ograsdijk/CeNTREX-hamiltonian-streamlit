import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import numpy.typing as npt
from centrex_tlf import hamiltonian, states

data_path = Path(__file__).parent / "data"


@dataclass
class HamiltonianX:
    uncoupled_states: list[states.UncoupledBasisState]
    coupled_states: list[states.CoupledBasisState]
    hamiltonian: hamiltonian.HamiltonianUncoupledX
    transform: npt.NDArray[np.complex_]


@dataclass
class HamiltonianB:
    coupled_states: list[states.CoupledBasisState]
    hamiltonian: hamiltonian.HamiltonianCoupledBOmega


def generate_hamiltonian_X(
    Evec: npt.NDArray[np.float_], Bvec: npt.NDArray[np.float_]
) -> tuple[
    list[states.CoupledBasisState],
    hamiltonian.reduced_hamiltonian.HamiltonianDiagonalized,
]:
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
    ham_diag = hamiltonian.generate_diagonalized_hamiltonian(ham, keep_order=True)
    return hamiltonian_x.coupled_states, ham_diag


def generate_hamiltonian_B(
    Evec: npt.NDArray[np.float_], Bvec: npt.NDArray[np.float_]
) -> tuple[
    list[states.CoupledBasisState],
    hamiltonian.reduced_hamiltonian.HamiltonianDiagonalized,
]:
    with open(data_path / "B_state_ham.pkl", "rb") as f:
        hamiltonian_b = HamiltonianB(**pickle.load(f))

    hamiltonian_func = hamiltonian.generate_coupled_hamiltonian_B_function(
        hamiltonian_b.hamiltonian
    )
    ham = hamiltonian_func(Evec, Bvec)
    ham_diag = hamiltonian.generate_diagonalized_hamiltonian(ham, keep_order=True)
    return hamiltonian_b.coupled_states, ham_diag


def generate_hamiltonian(
    electronic: states.ElectronicState,
    J: int,
    F1: float | Sequence[float] | npt.NDArray[np.floating],
    F: int | Sequence[int] | npt.NDArray[np.int_],
    E: float,
    B: float = 1e-3,
    P: Sequence[int] | int = [-1, 1],
) -> tuple[List[states.CoupledState], npt.NDArray[np.complex_]]:
    Evec = np.array([0, 0, E])
    Bvec = np.array([0, 0, B])

    if electronic == states.ElectronicState.X:
        selector = states.QuantumSelector(J=J, F1=F1, F=F, electronic=electronic)
        coupled_states, ham_diag = generate_hamiltonian_X(Evec, Bvec)
        states_select = [
            1 * state for state in states.generate_coupled_states_X(selector)
        ]
    elif electronic == states.ElectronicState.B:
        selector = states.QuantumSelector(J=J, F1=F1, F=F, electronic=electronic, P=P)
        coupled_states, ham_diag = generate_hamiltonian_B(Evec, Bvec)
        states_select = [
            1 * state for state in states.generate_coupled_states_B(selector)
        ]

    states_diag = hamiltonian.matrix_to_states(ham_diag.V, list(coupled_states))
    selected_states = states.find_exact_states(
        states_select,
        list(coupled_states),
        states_diag,
        ham_diag.H,
        ham_diag.V,
    )
    selected_states = [state.remove_small_components(1e-3) for state in selected_states]

    ham_reduced = hamiltonian.reduced_basis_hamiltonian(
        states_diag, ham_diag.H, selected_states
    )

    return selected_states, ham_reduced
