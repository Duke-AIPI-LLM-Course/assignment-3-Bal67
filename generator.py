from transformers import pipeline
from retriever import retrieve_best_chunk

llm_pipeline = pipeline("text-generation", model="gpt2", device="cpu")

def generate_response(query):
    try:
        context = retrieve_best_chunk(query)
        if not context:
            return "Sorry, I couldn't find relevant information."

        prompt = f"User Query: {query}\n\nAnswer:"

        response = llm_pipeline(
            prompt,
            max_length=400, 
            do_sample=False,
            temperature=0.3,  
            top_p=0.9,  
            pad_token_id=50256,  
            eos_token_id=50256  
        )

        if not response or "generated_text" not in response[0]:
            return "Sorry, the model failed to generate a response."

        return response[0]['generated_text'].strip().split(".")[0]+"."

    except Exception as e:
        return f"Error generating response: {str(e)}"

