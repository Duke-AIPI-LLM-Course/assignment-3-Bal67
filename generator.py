from transformers import pipeline
from retriever import retrieve_best_chunks
import torch

device = "cpu"  # Force CPU usage
torch.set_default_device("cpu")  # Explicitly set CPU as default

# Load a SMALL, RELIABLE model (GPT-2 or GPT-Neo)
llm_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=device)

def generate_response(query):
    try:
        context = retrieve_best_chunks(query)

        if not context:
            return "Sorry, I couldn't find relevant information."

        print(f"🔍 Final Prompt Sent to Model:\n{context}")

        # Force the model to return structured bullet points
        prompt = f"List the symptoms of diabetes clearly:\n\n{context}\n\nSymptoms:"

        response = llm_pipeline(
            prompt,
            max_length=150,
            do_sample=False,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=50256,
            eos_token_id=50256
        )

        if not response or "generated_text" not in response[0]:
            return "Sorry, the model failed to generate a response."

        return response[0]['generated_text'].strip()

    except Exception as e:
        return f"Error generating response: {str(e)}"

