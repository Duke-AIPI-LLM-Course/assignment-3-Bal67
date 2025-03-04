import streamlit as st
from generator import generate_response

st.title("🔎 RAG-Powered LLM on Diabetes")
st.write("Ask any question related to diabetes!")

query = st.text_input("Enter your question:")

if query:
    try:
        response = generate_response(query)

        if response.strip():
            st.write("Response:")
            st.write(response)
        else:
            st.write("No response was generated. Try a different question.")
    
    except Exception as e:
        st.write(f"⚠️ Error: {str(e)}")
