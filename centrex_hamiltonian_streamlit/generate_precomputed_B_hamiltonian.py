import pickle
from pathlib import Path

import numpy as np
from centrex_tlf import hamiltonian, states

data_path = Path(__file__).parent

Jmax = 6


# generate the hyperfine sublevels in J=0 to J=6
coupled_states = states.generate_coupled_states_excited(
    Js=np.arange(1, Jmax), Ps=[-1, 1], Omegas=1, basis=states.Basis.CoupledP
)

# generate the B hamiltonian terms
hamiltonian_coupled_terms = hamiltonian.generate_coupled_hamiltonian_B(coupled_states)

with open(data_path / "data/B_state_ham.pkl", "wb") as f:
    pickle.dump(
        dict(coupled_states=coupled_states, hamiltonian=hamiltonian_coupled_terms), f
    )
