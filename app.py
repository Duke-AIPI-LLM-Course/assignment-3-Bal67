import streamlit as st
from generator import generate_response

st.title("RAG-based LLM Application")
st.write("Ask me a question related to diabetes!")

# User input
query = st.text_input("Enter your query:")

if query:
    response = generate_response(query)
    st.write("Response:")
    st.write(response)
