import pickle
from pathlib import Path

import numpy as np
from centrex_tlf import hamiltonian, states

data_path = Path(__file__).parent

Jmax = 9

uncoupled_states = states.generate_uncoupled_states_ground(Js=np.arange(Jmax + 1))
coupled_states = states.generate_coupled_states_ground(Js=np.arange(Jmax + 1))

hamiltonian_uncoupled_terms = hamiltonian.generate_uncoupled_hamiltonian_X(
    uncoupled_states
)

transformation_matrix = hamiltonian.generate_transform_matrix(
    uncoupled_states, coupled_states
)


with open(data_path / "X_state_ham.pkl", "wb") as f:
    pickle.dump(
        dict(
            uncoupled_states=uncoupled_states,
            coupled_states=coupled_states,
            hamiltonian=hamiltonian_uncoupled_terms,
            transform=transformation_matrix,
        ),
        f,
    )
