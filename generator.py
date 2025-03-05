import asyncio
import transformers
import torch
from transformers import pipeline
from retriever import retrieve_best_chunks
import sys

# Fix asyncio event loop error in Streamlit
if sys.platform == "win32":  # Windows Fix
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load model with fallback to CPU
llm_pipeline = transformers.pipeline(
    task="text-generation",
    model="EleutherAI/gpt-neo-1.3B",
    device=0 if torch.cuda.is_available() else -1  # Uses GPU if available, otherwise CPU
)

def generate_response(query):
    try:
        # Retrieve context
        context = retrieve_best_chunks(query)

        if not context:
            return "Sorry, I couldn't find relevant information."

        # Create LLM input
        prompt = f"{context}\n\nAnswer:"

        # Ensure input isn't too long
        input_tokens = llm_pipeline.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if input_tokens.shape[1] > 200:  # Trim input to leave space for output
            prompt = llm_pipeline.tokenizer.decode(input_tokens[0, -200:], skip_special_tokens=True)

        # Debug: Print final model input
        print(f"🔍 Final Prompt Sent to Model:\n{prompt}")

        # Generate response with `max_new_tokens`
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

if __name__ == "__main__":
    print(generate_response("What is diabetes?"))
