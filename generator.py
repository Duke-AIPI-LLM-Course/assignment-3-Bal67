from transformers import pipeline
from retriever import retrieve_best_chunks
import torch

# Force CPU usage to prevent memory errors
device = "cpu"
torch.set_default_device("cpu")

# Load a reliable model
llm_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=device)

def generate_response(query):
    try:
        # Retrieve relevant context from the database
        context = retrieve_best_chunks(query)

        if not context:
            return "Sorry, I couldn't find relevant information."

        prompt = f"{context}\n\nAnswer:"


        input_tokens = llm_pipeline.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if input_tokens.shape[1] > 200:  
            prompt = llm_pipeline.tokenizer.decode(input_tokens[0, -200:], skip_special_tokens=True)


        print(f"Final Prompt Sent to Model:\n{prompt}")

        response = llm_pipeline(
            prompt,
            max_new_tokens=100,  # Allow model to generate up to 100 tokens
            do_sample=False,      # Ensure consistent responses
            temperature=0.3,      # Reduce randomness
            top_p=0.9,
            pad_token_id=50256,
            eos_token_id=50256
        )

        if not response or "generated_text" not in response[0]:
            return "Sorry, the model failed to generate a response."

        return response[0]['generated_text'].strip()

    except Exception as e:
        return f"Error generating response: {str(e)}"
