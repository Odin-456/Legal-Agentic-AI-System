import streamlit as st

from legal_agent import answer_question


st.set_page_config(
    page_title="Australian Property Law Assistant",
    page_icon="⚖️"
)


st.title(
    "Australian Property Law Assistant"
)

st.write(
    "Ask a question about NSW property law."
)

question = st.text_area(
    "Your question",
    placeholder=(
        "For example: "
        "My landlord wants to increase my rent. "
        "Can they do this?"
    ),
    height=120
)


if st.button("Ask Question"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching property-law documents..."
        ):

            answer, sources = answer_question(
                question
            )

        st.subheader("Answer")

        st.write(answer)

        if sources:

            st.subheader(
                "Legal sources"
            )

            for source in sources:

                metadata = source.get(
                    "metadata",
                    {}
                )

                st.write(
                    f"**{metadata.get('act', 'Unknown Act')}**"
                )

                st.write(
                    f"Section: "
                    f"{metadata.get('section', 'Unknown')}"
                )