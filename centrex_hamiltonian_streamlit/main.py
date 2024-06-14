import streamlit as st
from streamlit_navigation_bar import st_navbar

from centrex_hamiltonian_streamlit.pages import (
    show_level_scheme,
    show_transition_levels,
)


def main():
    navigation = {
        "Levels": show_level_scheme,
        "Transition Levels": show_transition_levels,
    }

    st.set_page_config(
        page_title="TlF Levels", initial_sidebar_state="collapsed", layout="wide"
    )

    page = st_navbar(["Levels", "Transition Levels"], options={"show_sidebar": False})

    navigation[page]()


if __name__ == "__main__":
    main()
