from transformers import pipeline
from retriever import retrieve_best_chunk

# Use a smaller model for better response generation
llm_pipeline = pipeline("text-generation", model="google/flan-t5-base")  

def generate_response(query):
    try:
        # Retrieve best document chunk
        context = retrieve_best_chunk(query)

        if not context:
            return "Sorry, I couldn't find relevant information."

        # Create prompt for LLM
        prompt = f"Context: {context}\n\nUser Query: {query}\n\nAnswer:"

        # Generate response using LLM
        response = llm_pipeline(prompt, max_length=100, do_sample=True)

        if not response or "generated_text" not in response[0]:
            return "Sorry, the model failed to generate a response."

        return response[0]['generated_text']

    except Exception as e:
        return f"Error generating response: {str(e)}"

