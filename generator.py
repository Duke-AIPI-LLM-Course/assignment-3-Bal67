import asyncio
import torch
from transformers import pipeline
from retriever import retrieve_best_chunks
import sys

llm_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    device="cpu"
)

def generate_response(query):
    try:
        context = retrieve_best_chunks(query)
        if not context:
            return "Sorry, I couldn't find relevant information."

        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        input_tokens = llm_pipeline.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if input_tokens.shape[1] > 200:
            prompt = llm_pipeline.tokenizer.decode(input_tokens[0, -200:], skip_special_tokens=True)

        response = llm_pipeline(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.7,   
            top_p=0.9,        
            repetition_penalty=2.5,  
            pad_token_id=50256,
            eos_token_id=50256
        )

        if not response or "generated_text" not in response[0]:
            return "Sorry, the model failed to generate a response."

        generated_text = response[0]['generated_text'].strip()

        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1].strip()

        return generated_text

    except Exception as e:
        return f"Error generating response: {str(e)}"

