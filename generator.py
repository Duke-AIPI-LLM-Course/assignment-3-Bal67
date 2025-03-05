import asyncio
import torch
from transformers import pipeline
from retriever import retrieve_best_chunks
import sys

# 🔹 Fix async issue in Streamlit
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 🔹 Load GPT-2 model with CPU (prevents crashes)
llm_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    device="cpu"
)

def generate_response(query):
    try:
        # 🔹 Retrieve relevant context from the retriever
        context = retrieve_best_chunks(query)

        if not context:
            return "Sorry, I couldn't find relevant information."

        # 🔹 Create a structured prompt to force GPT-2 to respond correctly
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        # 🔹 Trim input if too long
        input_tokens = llm_pipeline.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if input_tokens.shape[1] > 200:  # ✅ Prevents exceeding token limit
            prompt = llm_pipeline.tokenizer.decode(input_tokens[0, -200:], skip_special_tokens=True)

        # 🔹 Debugging: Show the actual input to the model
        print(f"🔍 Model Input:\n{prompt}")

        # 🔹 Generate response with max_new_tokens
        response = llm_pipeline(
            prompt,
            max_new_tokens=100,  # ✅ Allows for full responses
            do_sample=False,      # ✅ Keeps responses deterministic
            temperature=0.3,      # ✅ Reduces randomness
            top_p=0.9,            # ✅ Allows some variation but stays factual
            pad_token_id=50256,
            eos_token_id=50256
        )

        # 🔹 Ensure valid response
        if not response or "generated_text" not in response[0]:
            return "Sorry, the model failed to generate a response."

        # 🔹 Extract the generated answer
        generated_text = response[0]['generated_text'].strip()

        # 🔹 Remove everything before "Answer:" to keep only the model's response
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1].strip()

        return generated_text

    except Exception as e:
        return f"Error generating response: {str(e)}"

if __name__ == "__main__":
    print(generate_response("How many types of diabetes are there?"))
