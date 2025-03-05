import asyncio
import torch
from transformers import pipeline
from retriever import retrieve_best_chunks
import sys

llm_pipeline = pipeline(
    "text-generation",
    model="distilgpt2",
    device="cpu"
)

def generate_response(query):
    try:
        context = retrieve_best_chunks(query)
        if not context:
            return "Sorry, I couldn't find relevant information."

        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        response = llm_pipeline(
            prompt,
            max_length=500,  # Reduce length to fit Streamlit limits
            do_sample=True,   # Enable sampling for better responses
            temperature=0.7,  # Control creativity
            top_p=0.9         # Improve answer diversity
        )

        if not response or "generated_text" not in response[0]:
            return "Sorry, the model failed to generate a response."

        generated_text = response[0]['generated_text'].strip()

        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1].strip()

        return generated_text

    except Exception as e:
        return f"Error generating response: {str(e)}"

