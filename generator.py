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
            max_length=500,  
            do_sample=True,
            temperature=0.3,  
            top_p=0.5,
            repetition_penalty=2.0,
            num_return_sequences=1,
            pad_token_id=50256,
            eos_token_id=50256         
        )

        if not response or "generated_text" not in response[0]:
            return "Sorry, the model failed to generate a response."

        generated_text = response[0]["generated_text"].strip()
        sentences = generated_text.split(". ")  # Split into individual sentences

        return sentences[1] + "." if sentences else "No relevant answer found."

    except Exception as e:
        return f"Error generating response: {str(e)}"

