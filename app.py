import streamlit as st
from generator import generate_response

st.title("🔎 RAG-Powered LLM on Diabetes")
st.write("Ask any question related to diabetes!")

query = st.text_input("Enter your question:")

if query:
    response = generate_response(query)
    st.write("Response:")
    st.write(response)
