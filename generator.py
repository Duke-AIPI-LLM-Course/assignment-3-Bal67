from transformers import pipeline
from retriever import retrieve_best_chunk

llm_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct")

def generate_response(query):
    context = retrieve_best_chunk(query)
    prompt = f"Context: {context}\n\nUser Query: {query}\n\nAnswer:"
    response = llm_pipeline(prompt, max_length=100, do_sample=True)
    return response[0]['generated_text']

print(generate_response("How do I manage diabetes?"))
print(generate_response("What is the treatment for diabetes?"))
print(generate_response("What are the symptoms of diabetes?"))
print(generate_response("What causes diabetes?"))