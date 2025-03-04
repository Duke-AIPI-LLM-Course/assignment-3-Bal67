import streamlit as st
from generator import generate_response

st.title("🔎 RAG-Powered LLM on Diabetes")
st.write("Ask any question related to diabetes!")

query = st.text_input("Enter your question:")

if query:
    response = generate_response(query)

    # Prevent empty output
    if response.strip():
        st.write("Response:")
        st.write(response)
    else:
        st.write("No response was generated. Try a different question.")
