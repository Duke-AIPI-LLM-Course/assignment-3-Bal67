from transformers import pipeline
from retriever import retrieve_best_chunk

# Load open-source LLM
llm_pipeline = pipeline("text-generation", model="google/flan-t5-base")

def generate_response(query):
    context = retrieve_best_chunk(query)
    prompt = f"Context: {context}\n\nUser Query: {query}\n\nAnswer:"

    response = llm_pipeline(prompt, max_length=100, do_sample=True)
    return response[0]['generated_text']

# Test LLM response
print(generate_response("How do I manage diabetes?"))
